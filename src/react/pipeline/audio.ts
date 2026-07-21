/**
 * Audio pipeline primitives — the single implementation used by both the
 * standalone resolve path (`await vid.audio`) and the render pipeline.
 *
 * This module is the first brick of the shared pipeline layer: renderers
 * and resolvers call these functions instead of duplicating ffmpeg logic.
 */

import { groq } from "@ai-sdk/groq";
import { experimental_transcribe as transcribe } from "ai";
import { $ } from "bun";
import type { CacheStorage } from "../../ai-sdk/cache";
import { File } from "../../ai-sdk/file";
import type { FFmpegBackend } from "../../ai-sdk/providers/editly/backends";
import type { WordTiming } from "../../speech/types";
import { getResolveContext } from "../resolve-context";

// ---------------------------------------------------------------------------
// Extract audio from video (ffmpeg -vn)
// ---------------------------------------------------------------------------

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
    const result = await activeBackend.run({
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
    const data = await Bun.file(result.output.path).arrayBuffer();
    try {
      await Bun.file(result.output.path).delete?.();
    } catch {
      /* ignore cleanup errors */
    }
    return File.fromBuffer(new Uint8Array(data), "audio/mpeg");
  }

  // Fallback: local ffmpeg shell. Reads URLs directly (no full pre-download).
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
  const data = await Bun.file(outPath).arrayBuffer();
  try {
    await Bun.file(outPath).delete?.();
  } catch {
    /* ignore cleanup errors */
  }
  return File.fromBuffer(new Uint8Array(data), "audio/mpeg");
}

// ---------------------------------------------------------------------------
// Silence detection (ffmpeg silencedetect)
// ---------------------------------------------------------------------------

export interface SilenceDetectOptions {
  /** Noise threshold in dB — audio below this level counts as silence. Default -30. */
  noiseDb?: number;
  /** Minimum silence duration in seconds to report. Default 0.3. */
  minDuration?: number;
}

export interface TimeRange {
  start: number;
  end: number;
}

/**
 * Detect intervals of silence in an audio file via ffmpeg `silencedetect`.
 *
 * Note: silencedetect finds *sound*, not speech — ambient noise and
 * footsteps count as sound. For speech-specific boundaries combine with
 * `transcribeAudio()` word timings.
 *
 * Runs local ffmpeg only (silencedetect output is on stderr, which cloud
 * backends don't return). Reads URLs directly when available.
 */
export async function detectSilence(
  file: File,
  options: SilenceDetectOptions = {},
): Promise<TimeRange[]> {
  const noiseDb = options.noiseDb ?? -30;
  const minDuration = options.minDuration ?? 0.3;
  const input = file.url ?? (await file.toTempFile());

  const result =
    await $`ffmpeg -i ${input} -af silencedetect=noise=${noiseDb}dB:d=${minDuration} -f null -`
      .quiet()
      .nothrow();
  // silencedetect logs to stderr; ffmpeg exits 0 on success for -f null
  const stderr = result.stderr.toString();

  const ranges: TimeRange[] = [];
  let currentStart: number | undefined;
  for (const line of stderr.split("\n")) {
    const startMatch = line.match(/silence_start:\s*([\d.]+)/);
    if (startMatch?.[1]) {
      currentStart = Number.parseFloat(startMatch[1]);
      continue;
    }
    const endMatch = line.match(/silence_end:\s*([\d.]+)/);
    if (endMatch?.[1] && currentStart !== undefined) {
      ranges.push({
        start: currentStart,
        end: Number.parseFloat(endMatch[1]),
      });
      currentStart = undefined;
    }
  }
  // Trailing silence (silence_start without silence_end — runs to EOF)
  if (currentStart !== undefined) {
    const durationMatch = stderr.match(/Duration:\s*(\d+):(\d+):([\d.]+)/);
    if (durationMatch) {
      const total =
        Number.parseInt(durationMatch[1] ?? "0", 10) * 3600 +
        Number.parseInt(durationMatch[2] ?? "0", 10) * 60 +
        Number.parseFloat(durationMatch[3] ?? "0");
      ranges.push({ start: currentStart, end: total });
    }
  }
  return ranges;
}

/**
 * Compute the bounds of audible content: `start` of the first sound and
 * `end` of the last sound, derived from silence intervals.
 *
 * @param duration Total media duration in seconds.
 */
export function computeSoundBounds(
  silences: TimeRange[],
  duration: number,
): TimeRange {
  let start = 0;
  let end = duration;
  for (const s of silences) {
    // Leading silence
    if (s.start <= 0.05 && s.end > start) start = s.end;
    // Trailing silence
    if (duration > 0 && s.end >= duration - 0.05 && s.start < end)
      end = s.start;
  }
  return { start, end: Math.max(start, end) };
}

// ---------------------------------------------------------------------------
// Transcription (Groq Whisper word-level, cached)
// ---------------------------------------------------------------------------

export interface TranscriptionResult {
  text: string;
  words: WordTiming[];
}

interface GroqWordShape {
  word: string;
  start: number;
  end: number;
}

/** Stable identity for audio content used in transcription cache keys.
 *  Content hash — upload URLs contain Date.now()+random and would make
 *  keys non-deterministic (same reasoning as computeCacheKey). */
function contentCacheIdentity(bytes: Uint8Array): string {
  return `h${Bun.hash(bytes).toString(36)}:${bytes.byteLength}`;
}

/**
 * Transcribe an audio file to `{ text, words }` with word-level timestamps.
 *
 * Uses the gateway transcription model from render defaults when available,
 * falling back to direct Groq `whisper-large-v3`. Results are cached by
 * audio content hash in the active cache storage.
 */
export async function transcribeAudio(
  file: File,
  options: {
    cache?: CacheStorage;
    model?: Parameters<typeof transcribe>[0]["model"];
  } = {},
): Promise<TranscriptionResult> {
  const ctx = getResolveContext();
  const cache = options.cache ?? ctx?.cache;
  const bytes = await file.arrayBuffer();
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);

  const model = options.model ?? groq.transcription("whisper-large-v3");
  const modelId =
    typeof model === "string"
      ? model
      : ((model as { modelId?: string }).modelId ?? "whisper");
  const cacheKey = `transcribeAudio:${modelId}:${contentCacheIdentity(data)}`;

  if (cache) {
    const cached = await cache.get(cacheKey);
    if (cached !== undefined) return cached as TranscriptionResult;
  }

  const isGatewayModel = options.model !== undefined;
  const result = await transcribe({
    model,
    audio: data,
    providerOptions: isGatewayModel
      ? {}
      : {
          groq: {
            responseFormat: "verbose_json",
            timestampGranularities: ["word"],
          },
        },
  });

  // Extract words: from providerMetadata (gateway) or response body (direct groq)
  let words: WordTiming[] = [];
  const metaWords = (
    result.providerMetadata?.varg as { words?: GroqWordShape[] } | undefined
  )?.words;
  if (metaWords && metaWords.length > 0) {
    words = metaWords.map((w) => ({
      word: w.word,
      start: w.start,
      end: w.end,
    }));
  } else {
    const rawBody = (result.responses[0] as { body?: unknown })?.body;
    const bodyWords = (rawBody as { words?: GroqWordShape[] } | undefined)
      ?.words;
    if (Array.isArray(bodyWords)) {
      words = bodyWords
        .filter(
          (w) =>
            typeof w?.word === "string" &&
            typeof w?.start === "number" &&
            typeof w?.end === "number",
        )
        .map((w) => ({ word: w.word, start: w.start, end: w.end }));
    }
  }

  const transcription: TranscriptionResult = { text: result.text, words };
  if (cache) {
    await cache.set(cacheKey, transcription);
  }
  return transcription;
}
