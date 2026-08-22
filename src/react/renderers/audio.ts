import type { FFmpegBackend } from "../../ai-sdk/providers/editly/backends";

interface ResolveVideoMixVolumeOptions {
  backend: FFmpegBackend;
  keepAudio?: boolean;
  path: string;
  volume?: number;
}

/** Warn at most once per source, so a 13-clip render emits 13 lines, not 13x N. */
const warned = new Set<string>();

function warnOnce(key: string, message: string): void {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(message);
}

/** Trim long signed URLs down to something readable in a log line. */
function shortPath(path: string): string {
  const withoutQuery = path.split("?")[0] ?? path;
  const tail = withoutQuery.split("/").pop();
  return tail && tail.length > 0 ? tail : withoutQuery;
}

/**
 * Decide how loud a video layer's own audio should be in the mix.
 *
 * Source audio is on by default, but only when the input actually has an audio
 * stream — referencing `[n:a]` for a stream that does not exist is a hard
 * ffmpeg failure that would take down the whole render.
 *
 * `keepAudio: true` forces it on without probing, `keepAudio: false` forces it
 * off. When the backend cannot tell us whether audio exists we have to mute,
 * because guessing wrong crashes the render — but we say so out loud rather
 * than silently dropping a track the caller paid to generate.
 */
export async function resolveVideoMixVolume({
  backend,
  keepAudio,
  path,
  volume,
}: ResolveVideoMixVolumeOptions): Promise<number> {
  if (keepAudio === false) return 0;
  if (keepAudio === true) return volume ?? 1;

  let info: Awaited<ReturnType<FFmpegBackend["ffprobe"]>>;
  try {
    info = await backend.ffprobe(path);
  } catch (error) {
    warnOnce(
      `probe-failed:${path}`,
      `[varg] Could not probe "${shortPath(path)}" for an audio track ` +
        `(${error instanceof Error ? error.message : String(error)}). ` +
        "Muting it. Pass keepAudio: true to force its audio into the mix.",
    );
    return 0;
  }

  if (info.hasAudio === true) return volume ?? 1;
  if (info.hasAudio === false) return 0;

  // Backend resolved the file but does not report stream-level metadata.
  warnOnce(
    `no-stream-info:${backend.name}`,
    `[varg] Backend "${backend.name}" does not report whether a video has an ` +
      "audio track, so source audio is being muted " +
      `(first seen on "${shortPath(path)}"). ` +
      "Pass keepAudio: true on the Video to force its audio into the mix.",
  );
  return 0;
}

/** Test-only: clear the warn-once cache between cases. */
export function resetAudioWarnings(): void {
  warned.clear();
}
