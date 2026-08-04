import { unlinkSync } from "node:fs";
import type { ImageModelV3File } from "@ai-sdk/provider";
import type { StorageProvider } from "./storage/types";

// ---------------------------------------------------------------------------
// Temp-file ownership
//
// `toTempFile()` materializes a buffer-backed File to disk so ffmpeg (which
// only takes paths) can read it. Previously each call minted a fresh
// `varg-<ts>-<rand>` path, so one File probed by ffprobe, then silencedetect,
// then bandpassed silencedetect wrote its bytes three times — and cleanup was
// the *caller's* job, done inconsistently (detectSilence/detectSpeechActivity
// deleted in `finally`; probeDurationLocal never did, leaking one file per
// clip).
//
// The two problems are one problem: materialization can only be memoized if
// the File — not the caller — owns the path's lifetime. So the File keeps the
// path, and it is released when the File becomes unreachable (FinalizationRegistry)
// or when the process exits, whichever comes first.
// ---------------------------------------------------------------------------

/** Paths owned by live Files, swept on process exit. */
const liveTempPaths = new Set<string>();

function unlinkQuiet(path: string): void {
  try {
    unlinkSync(path);
  } catch {
    /* already gone, or never written */
  }
}

function releaseTempPath(path: string): void {
  if (!liveTempPaths.delete(path)) return;
  unlinkQuiet(path);
}

/**
 * Reclaim a File's temp file once the File itself is unreachable. Holds only
 * the path string — capturing the File would keep it alive forever.
 */
const tempFileFinalizer =
  typeof FinalizationRegistry === "function"
    ? new FinalizationRegistry<string>(releaseTempPath)
    : undefined;

let exitHookInstalled = false;

/** Sweep any still-live temp files on normal process exit (sync — `exit` allows no async). */
function installExitHook(): void {
  if (exitHookInstalled) return;
  exitHookInstalled = true;
  process.on("exit", () => {
    for (const path of liveTempPaths) unlinkQuiet(path);
    liveTempPaths.clear();
  });
}

/** Type of generated content */
export type GeneratedFileType =
  | "image"
  | "video"
  | "speech"
  | "music"
  | "captions";

/** Metadata for AI-generated files */
export interface FileMetadata {
  /** Type of generated content */
  type?: GeneratedFileType;
  /** Model used to generate */
  model?: string;
  /** Original prompt used */
  prompt?: string;
  /** Duration in seconds (for video/audio files) */
  duration?: number;
}

export class File {
  private _data: Uint8Array | null = null;
  private _url: string | null = null;
  private _mediaType: string;
  private _loader: (() => Promise<Uint8Array>) | null = null;
  private _metadata: FileMetadata = {};
  /** Materialized temp path, once `toTempFile()` has written one. */
  private _tempPath: string | null = null;
  /** In-flight materialization — dedupes concurrent `toTempFile()` calls. */
  private _tempPromise: Promise<string> | null = null;

  private constructor(
    options:
      | { data: Uint8Array; mediaType: string; url?: string }
      | { url: string; mediaType?: string }
      | { loader: () => Promise<Uint8Array>; mediaType: string },
  ) {
    if ("data" in options) {
      this._data = options.data;
      this._mediaType = options.mediaType;
      this._url = options.url ?? null;
    } else if ("url" in options) {
      this._url = options.url;
      this._mediaType = options.mediaType ?? inferMediaType(options.url);
    } else {
      this._loader = options.loader;
      this._mediaType = options.mediaType;
    }
  }

  static fromPath(path: string, mediaType?: string): File {
    const resolvedMediaType = mediaType ?? inferMediaType(path);
    return new File({
      loader: async () => {
        const file = Bun.file(path);
        return new Uint8Array(await file.arrayBuffer());
      },
      mediaType: resolvedMediaType,
    });
  }

  static fromUrl(url: string, mediaType?: string): File {
    return new File(mediaType != null ? { url, mediaType } : { url });
  }

  /** Hydrate a File from the render service response shape */
  static fromRenderFile(input: {
    url: string | null;
    mediaType: string;
    metadata?: FileMetadata;
  }): File | null {
    if (!input.url) return null;
    const file = File.fromUrl(input.url, input.mediaType);
    if (input.metadata) file.withMetadata(input.metadata);
    return file;
  }

  static fromBuffer(data: Uint8Array, mediaType: string): File {
    return new File({ data, mediaType });
  }

  static fromGenerated(generated: {
    uint8Array: Uint8Array;
    mediaType: string;
    url?: string;
  }): File {
    return new File({
      data: generated.uint8Array,
      mediaType: generated.mediaType,
      ...(generated.url != null && { url: generated.url }),
    });
  }

  static fromArrayBuffer(buffer: ArrayBuffer, mediaType: string): File {
    return new File({ data: new Uint8Array(buffer), mediaType });
  }

  static async fromBlob(blob: Blob, mediaType?: string): Promise<File> {
    const data = new Uint8Array(await blob.arrayBuffer());
    return new File({ data, mediaType: mediaType ?? blob.type });
  }

  static from(input: {
    uint8Array: Uint8Array;
    mimeType?: string;
    mediaType?: string;
  }): File;
  static from(
    input: string | Uint8Array | ArrayBuffer | Blob,
    mediaType?: string,
  ): File | Promise<File>;
  static from(
    input:
      | string
      | Uint8Array
      | ArrayBuffer
      | Blob
      | { uint8Array: Uint8Array; mimeType?: string; mediaType?: string },
    mediaType?: string,
  ): File | Promise<File> {
    if (typeof input === "object" && input !== null && "uint8Array" in input) {
      const mime =
        input.mimeType ?? input.mediaType ?? "application/octet-stream";
      return File.fromBuffer(input.uint8Array, mime);
    }
    if (typeof input === "string" && /^https?:\/\//.test(input)) {
      return File.fromUrl(input, mediaType);
    }
    if (typeof input === "string") {
      return File.fromPath(input, mediaType);
    }
    if (input instanceof Blob) {
      return File.fromBlob(input, mediaType);
    }
    if (input instanceof ArrayBuffer) {
      return File.fromArrayBuffer(
        input,
        mediaType ?? "application/octet-stream",
      );
    }
    return File.fromBuffer(input, mediaType ?? "application/octet-stream");
  }

  get mediaType(): string {
    return this._mediaType;
  }

  isImage(): boolean {
    return this._mediaType.startsWith("image/");
  }

  isAudio(): boolean {
    return this._mediaType.startsWith("audio/");
  }

  isVideo(): boolean {
    return this._mediaType.startsWith("video/");
  }

  get url(): string | null {
    return this._url;
  }

  /** Get file metadata (type, model, prompt, duration) */
  get metadata(): FileMetadata {
    return this._metadata;
  }

  /** Duration in seconds (for video/audio files) */
  get duration(): number | undefined {
    return this._metadata.duration;
  }

  /** Set metadata and return this for chaining */
  withMetadata(metadata: FileMetadata): this {
    this._metadata = { ...this._metadata, ...metadata };
    return this;
  }

  async data(): Promise<Uint8Array> {
    if (this._data) return this._data;
    if (this._loader) {
      this._data = await this._loader();
      return this._data;
    }
    if (this._url) {
      const response = await fetch(this._url);
      this._data = new Uint8Array(await response.arrayBuffer());
      return this._data;
    }
    throw new Error("File has no data source");
  }

  async arrayBuffer(): Promise<Uint8Array> {
    return this.data();
  }

  async blob(): Promise<Blob> {
    const data = await this.arrayBuffer();
    return new Blob([data as BlobPart], { type: this._mediaType });
  }

  /**
   * Upload file to storage and return the URL. Returns cached URL if already uploaded.
   * @param storage - Storage provider to use for upload
   * @returns URL of the uploaded file
   */
  async upload(storage: StorageProvider): Promise<string> {
    if (this._url) return this._url;
    const data = await this.data();
    const key = `varg/${Date.now()}-${Math.random().toString(36).slice(2)}${this.extensionFromMediaType()}`;
    this._url = await storage.upload(data, key, this._mediaType);
    return this._url;
  }

  async base64(): Promise<string> {
    const data = await this.arrayBuffer();
    let binary = "";
    for (const byte of data) {
      binary += String.fromCharCode(byte);
    }
    return btoa(binary);
  }

  toJSON(): { url: string | null; mediaType: string; metadata: FileMetadata } {
    return {
      url: this._url,
      mediaType: this._mediaType,
      metadata: this._metadata,
    };
  }

  async toInput(): Promise<ImageModelV3File> {
    if (this._url) {
      return { type: "url", url: this._url };
    }
    const data = await this.arrayBuffer();
    return { type: "file", mediaType: this._mediaType, data };
  }

  /**
   * Materialize this file's bytes to a temp path on disk, for tools that
   * only accept paths (ffmpeg/ffprobe).
   *
   * Memoized: repeated calls on the same File return the same path and write
   * once. A single clip is probed for duration, then run through
   * `silencedetect`, then through bandpassed `silencedetect` — three
   * consumers, one file.
   *
   * The File owns the path. It is unlinked when the File is garbage-collected
   * or the process exits; callers must NOT delete it, or the next consumer
   * gets a path to nothing. Use {@link releaseTempFile} to free it eagerly
   * when you know the File is done.
   *
   * @returns Path to the temporary file
   */
  async toTempFile(): Promise<string> {
    if (this._tempPath !== null) {
      // Re-materialize if something outside deleted the file (a stale caller,
      // a tmp reaper). Cheaper than making every consumer re-verify.
      if (await Bun.file(this._tempPath).exists()) return this._tempPath;
      liveTempPaths.delete(this._tempPath);
      this._tempPath = null;
    }
    if (this._tempPromise) return this._tempPromise;

    this._tempPromise = (async () => {
      const data = await this.data();
      const ext = this.extensionFromMediaType();
      const tmpDir = process.env.TMPDIR ?? "/tmp";
      const filename = `varg-${Date.now()}-${Math.random().toString(36).slice(2)}${ext}`;
      const path = `${tmpDir}/${filename}`;
      await Bun.write(path, data);
      this._tempPath = path;
      liveTempPaths.add(path);
      installExitHook();
      tempFileFinalizer?.register(this, path, this);
      return path;
    })();

    try {
      return await this._tempPromise;
    } finally {
      this._tempPromise = null;
    }
  }

  /**
   * Delete this File's temp file now, instead of waiting for GC or exit.
   *
   * Optional — an unreleased temp file is reclaimed either way. Worth calling
   * in long-lived loops over large media, where waiting for GC means holding
   * gigabytes of /tmp. A later `toTempFile()` simply re-materializes.
   */
  releaseTempFile(): void {
    if (this._tempPath === null) return;
    tempFileFinalizer?.unregister(this);
    releaseTempPath(this._tempPath);
    this._tempPath = null;
  }

  static async toTemp(
    file:
      | { uint8Array: Uint8Array; mimeType?: string; mediaType?: string }
      | File,
  ): Promise<string> {
    if (file instanceof File) {
      return file.toTempFile();
    }
    const f = File.from(file);
    return f.toTempFile();
  }

  private extensionFromMediaType(): string {
    const extMap: Record<string, string> = {
      "image/png": ".png",
      "image/jpeg": ".jpg",
      "image/gif": ".gif",
      "image/webp": ".webp",
      "audio/mpeg": ".mp3",
      "audio/wav": ".wav",
      "audio/mp4": ".m4a",
      "video/mp4": ".mp4",
      "video/webm": ".webm",
      "video/quicktime": ".mov",
      "text/x-ssa": ".ass",
      "application/x-subrip": ".srt",
    };
    return extMap[this._mediaType] ?? "";
  }
}

function inferMediaType(path: string): string {
  const ext = path.split(".").pop()?.toLowerCase();
  const mimeTypes: Record<string, string> = {
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    gif: "image/gif",
    webp: "image/webp",
    svg: "image/svg+xml",
    mp3: "audio/mpeg",
    wav: "audio/wav",
    ogg: "audio/ogg",
    m4a: "audio/mp4",
    aac: "audio/aac",
    mp4: "video/mp4",
    webm: "video/webm",
    mov: "video/quicktime",
    ass: "text/x-ssa",
    ssa: "text/x-ssa",
    srt: "application/x-subrip",
  };
  return mimeTypes[ext ?? ""] ?? "application/octet-stream";
}

export function files(...paths: string[]): File[] {
  return paths.map((p) => File.fromPath(p));
}
