/**
 * Running `silencedetect` through a cloud ffmpeg backend.
 *
 * `silencedetect` prints its findings to **stderr**, and cloud backends
 * return output *files*, not stderr — so the local implementation's
 * "read stderr" step has no cloud equivalent. That is a limitation of how
 * we read the result, not of the backend: Rendi runs any ffmpeg command,
 * `silencedetect` included.
 *
 * ffmpeg's `ametadata=mode=print:file=…` writes the same detections to a
 * file, which every backend does return. So the cloud path appends
 * `ametadata` to the filter chain and parses the file it produces.
 *
 * The two formats are NOT interchangeable:
 *
 *   stderr     [silencedetect @ 0x…] silence_start: 0
 *   ametadata  lavfi.silence_start=0
 *
 * hence `parseSilenceMetadata` next to `parseSilenceRanges` rather than one
 * regex stretched over both.
 */

import type { File } from "../../ai-sdk/file";
import type { FFmpegBackend } from "../../ai-sdk/providers/editly/backends";
import type { TimeRange } from "./silence";

/**
 * Parse ffmpeg `ametadata=mode=print` output into TimeRange[].
 *
 * The file interleaves frame headers with `lavfi.*` key/value lines:
 *
 *   frame:375  pts:384000  pts_time:8
 *   lavfi.silence_start=7.70831
 *   lavfi.silence_end=8.10385
 *   lavfi.silence_duration=0.395542
 *
 * A file that ends silent emits a final `silence_start` with no matching
 * `silence_end`. The stderr parser recovers that end from ffmpeg's
 * `Duration:` banner, but the metadata file has no such line — which is why
 * `duration` is a required parameter here rather than an optional hint.
 */
export function parseSilenceMetadata(
  text: string,
  duration: number,
): TimeRange[] {
  const ranges: TimeRange[] = [];
  let currentStart: number | undefined;

  for (const line of text.split("\n")) {
    const startMatch = line.match(/lavfi\.silence_start=\s*(-?[\d.]+)/);
    if (startMatch?.[1]) {
      // ffmpeg can report a tiny negative start when silence begins on the
      // very first frame; clamp so downstream range math stays sane.
      currentStart = Math.max(0, Number.parseFloat(startMatch[1]));
      continue;
    }
    const endMatch = line.match(/lavfi\.silence_end=\s*([\d.]+)/);
    if (endMatch?.[1] && currentStart !== undefined) {
      ranges.push({
        start: currentStart,
        end: Number.parseFloat(endMatch[1]),
      });
      currentStart = undefined;
    }
  }

  // Trailing silence: runs to EOF, so the caller-supplied duration is the end.
  if (currentStart !== undefined && duration > currentStart) {
    ranges.push({ start: currentStart, end: duration });
  }
  return ranges;
}

/**
 * Run an audio filter chain ending in `silencedetect` through `backend` and
 * return the detected silences.
 *
 * `audioFilter` must be the chain WITHOUT the trailing `ametadata` — this
 * appends it, pointed at the command's output file.
 *
 * @param duration Media duration, for the trailing-silence bound (see
 *        `parseSilenceMetadata`). Callers that already know it should pass
 *        it; `detectSilence` probes for it.
 */
export async function detectSilenceViaBackend(
  file: File,
  backend: FFmpegBackend,
  audioFilter: string,
  duration: number,
): Promise<TimeRange[]> {
  const outputPath = `silence-${Date.now()}-${Math.random().toString(36).slice(2)}.txt`;

  // `{{out_1}}` is the backend's own placeholder for the output file; the
  // local backend substitutes a path, Rendi an upload target. Writing the
  // metadata there is what makes the result retrievable at all.
  const result = await backend.run({
    inputs: [{ path: file }],
    outputArgs: [
      "-af",
      `${audioFilter},ametadata=mode=print:file={{out_1}}`,
      "-f",
      "null",
    ],
    outputPath,
  });

  const text =
    result.output.type === "url"
      ? await fetchText(result.output.url)
      : await Bun.file(result.output.path).text();

  return parseSilenceMetadata(text, duration);
}

async function fetchText(url: string): Promise<string> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch silence metadata (${response.status}) from ${url}`,
    );
  }
  return response.text();
}
