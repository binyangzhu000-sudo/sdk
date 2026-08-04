/**
 * Audio extraction — pull the audio track out of a video file as MP3.
 *
 * Used by both the standalone resolve path (`await vid.audio`) and the
 * render pipeline (`renderAudio` → `runStep`).
 */

import { $ } from "bun";
import { File } from "../../ai-sdk/file";
import type { FFmpegBackend } from "../../ai-sdk/providers/editly/backends";
import { getResolveContext } from "../resolve-context";

/**
 * Extract the audio track from a video file as MP3.
 *
 * Routes through the FFmpegBackend when available (local or cloud/Rendi),
 * falling back to a direct local `ffmpeg` shell command (top-level `await`
 * outside render()).
 *
 * @throws Error when the video has no audio track.
 */
export async function extractAudio(
  file: File,
  backend?: FFmpegBackend,
): Promise<File> {
  const ctx = getResolveContext();
  const activeBackend = backend ?? ctx?.backend;
  const suffix = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
  const outPath = `/tmp/varg-audio-${suffix}.mp3`;

  if (activeBackend) {
    return extractViaBackend(file, activeBackend, outPath);
  }

  // Fallback: local ffmpeg shell. Reads URLs directly (no full pre-download);
  // for non-URL inputs a temp file is created and cleaned up in finally.
  return extractViaLocalFfmpeg(file, outPath);
}

async function extractViaBackend(
  file: File,
  backend: FFmpegBackend,
  outPath: string,
): Promise<File> {
  // Pass the File through for non-URL inputs: the backend's resolvePath calls
  // toTempFile(), which is memoized on the File, so the source video is
  // materialized once and reused by any later consumer instead of being
  // written here and deleted out from under them.
  const result = await backend.run({
    inputs: [{ path: file }],
    outputArgs: ["-vn", "-acodec", "libmp3lame", "-q:a", "2"],
    outputPath: outPath,
  });
  if (result.output.type === "url") {
    const response = await fetch(result.output.url);
    const bytes = new Uint8Array(await response.arrayBuffer());
    return File.fromGenerated({
      uint8Array: bytes,
      mediaType: "audio/mpeg",
      url: result.output.url,
    });
  }
  // `outPath` is this function's own output, not a File-owned temp — ours to delete.
  const data = await Bun.file(result.output.path).arrayBuffer();
  try {
    await Bun.file(result.output.path).delete?.();
  } catch {
    /* ignore cleanup errors */
  }
  return File.fromBuffer(new Uint8Array(data), "audio/mpeg");
}

async function extractViaLocalFfmpeg(
  file: File,
  outPath: string,
): Promise<File> {
  // Temp input (if any) is owned by `file` — memoized and shared downstream.
  const input = file.url ?? (await file.toTempFile());
  const result =
    await $`ffmpeg -y -i ${input} -vn -acodec libmp3lame -q:a 2 ${outPath}`
      .quiet()
      .nothrow();
  if (result.exitCode !== 0) {
    const stderr = result.stderr.toString().trim();
    throw new Error(
      `ffmpeg audio extraction failed (exit ${result.exitCode}): ${stderr || "unknown error"}`,
    );
  }
  // `outPath` is our own output — ours to delete.
  const data = await Bun.file(outPath).arrayBuffer();
  try {
    await Bun.file(outPath).delete?.();
  } catch {
    /* ignore cleanup errors */
  }
  return File.fromBuffer(new Uint8Array(data), "audio/mpeg");
}
