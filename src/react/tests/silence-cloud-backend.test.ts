/**
 * Silence/speech detection on cloud ffmpeg backends.
 *
 * `detectSilence` and `detectSpeechActivity` used to throw outright on any
 * backend whose name wasn't "local", with the message "cloud backend does
 * not support silencedetect". That reason was wrong: Rendi runs any ffmpeg
 * command. What it does not return is **stderr**, which is where
 * `silencedetect` prints — so the limitation was in how we read the result.
 *
 * The throw was worse than an error, because `AudioNode.transcribe()` wraps
 * these calls in `try { … } catch { return raw }`. Inside a cloud render the
 * exception was swallowed and `speechRange()` silently returned unrefined
 * whisper timings — the exact bad boundaries the feature exists to fix.
 *
 * The fix routes cloud backends through `ametadata=mode=print:file=…`, whose
 * output file every backend does return. Fixtures below are real ffmpeg
 * output: the metadata sample was captured from a live Rendi run
 * (popeye_talking.mp4), the local one from ffmpeg's stderr.
 */

import { describe, expect, test } from "bun:test";
import { File } from "../../ai-sdk/file";
import type { FFmpegBackend } from "../../ai-sdk/providers/editly/backends";
import { detectSilence, parseSilenceRanges } from "../primitives/silence";
import { parseSilenceMetadata } from "../primitives/silence-backend";
import { detectSpeechActivity } from "../primitives/speech-activity";
import { withResolveContext } from "../resolve-context";

/** Verbatim from a live Rendi run of silencedetect+ametadata. */
const RENDI_METADATA = `frame:375  pts:384000  pts_time:8
lavfi.silence_start=7.70831
frame:379  pts:388096  pts_time:8.08533
lavfi.silence_end=8.10385
lavfi.silence_duration=0.395542
frame:516  pts:528384  pts_time:11.008
lavfi.silence_start=10.7127
frame:519  pts:531456  pts_time:11.072
lavfi.silence_end=11.0909
lavfi.silence_duration=0.378146
`;

/** A clip that ends silent: last silence_start has no matching end. */
const TRAILING_METADATA = `frame:40  pts:40960  pts_time:0.928
lavfi.silence_start=0.999909
`;

/**
 * Records what the backend was asked to run, and replies with a metadata
 * file — mirroring Rendi, which returns output files as URLs.
 */
function makeCloudBackend(metadata: string, duration = 20) {
  const calls: { outputArgs: string[]; outputPath: string }[] = [];
  const backend = {
    name: "rendi",
    async ffprobe() {
      return { duration, width: 0, height: 0 };
    },
    async resolvePath() {
      return "https://storage.example/in.mp3";
    },
    async run(options: { outputArgs?: string[]; outputPath: string }) {
      calls.push({
        outputArgs: options.outputArgs ?? [],
        outputPath: options.outputPath,
      });
      // Rendi hands back a storage URL; a data: URL keeps the test offline
      // while still exercising the fetch branch.
      return {
        output: {
          type: "url" as const,
          url: `data:text/plain;base64,${Buffer.from(metadata).toString("base64")}`,
        },
      };
    },
  } as unknown as FFmpegBackend;
  return { backend, calls };
}

const audio = () => File.fromBuffer(new Uint8Array([1, 2, 3]), "audio/mpeg");

describe("parseSilenceMetadata", () => {
  test("parses lavfi key=value pairs from a real Rendi run", () => {
    expect(parseSilenceMetadata(RENDI_METADATA, 20)).toEqual([
      { start: 7.70831, end: 8.10385 },
      { start: 10.7127, end: 11.0909 },
    ]);
  });

  test("closes a trailing silence with the supplied duration", () => {
    // The metadata file carries no `Duration:` banner, so the caller's
    // duration is the only way to bound a clip that ends silent.
    expect(parseSilenceMetadata(TRAILING_METADATA, 2)).toEqual([
      { start: 0.999909, end: 2 },
    ]);
  });

  test("drops a trailing silence that starts past the duration", () => {
    expect(parseSilenceMetadata(TRAILING_METADATA, 0.5)).toEqual([]);
  });

  test("clamps a negative start to zero", () => {
    const ranges = parseSilenceMetadata("lavfi.silence_start=-0.002\n", 1);
    expect(ranges[0]!.start).toBe(0);
  });

  test("returns nothing for output with no detections", () => {
    expect(parseSilenceMetadata("frame:1  pts:0  pts_time:0\n", 5)).toEqual([]);
  });

  test("is a distinct format from the stderr parser", () => {
    // Neither parser understands the other's output — the reason both exist.
    expect(parseSilenceRanges(RENDI_METADATA)).toEqual([]);
    expect(
      parseSilenceMetadata(
        "[silencedetect @ 0x7f] silence_start: 1\n[silencedetect @ 0x7f] silence_end: 2\n",
        5,
      ),
    ).toEqual([]);
  });
});

describe("detectSilence on a cloud backend", () => {
  test("returns ranges instead of throwing", async () => {
    const { backend } = makeCloudBackend(RENDI_METADATA);
    const ranges = await withResolveContext({ backend }, () =>
      detectSilence(audio()),
    );
    expect(ranges).toEqual([
      { start: 7.70831, end: 8.10385 },
      { start: 10.7127, end: 11.0909 },
    ]);
  });

  test("asks ffmpeg to write metadata to the output file", async () => {
    const { backend, calls } = makeCloudBackend(RENDI_METADATA);
    await withResolveContext({ backend }, () =>
      detectSilence(audio(), { noiseDb: -25, minDuration: 0.5 }),
    );

    const args = calls[0]!.outputArgs.join(" ");
    expect(args).toContain("silencedetect=noise=-25dB:d=0.5");
    // Without ametadata the detections would only reach stderr, which cloud
    // backends do not return — this is the whole mechanism.
    expect(args).toContain("ametadata=mode=print:file={{out_1}}");
    expect(args).toContain("-f null");
    expect(calls[0]!.outputPath.endsWith(".txt")).toBe(true);
  });

  test("probes for duration to close a trailing silence", async () => {
    const { backend } = makeCloudBackend(TRAILING_METADATA, 2);
    const ranges = await withResolveContext({ backend }, () =>
      detectSilence(audio()),
    );
    expect(ranges).toEqual([{ start: 0.999909, end: 2 }]);
  });
});

describe("detectSpeechActivity on a cloud backend", () => {
  test("returns activity instead of throwing", async () => {
    const { backend } = makeCloudBackend(RENDI_METADATA, 20);
    const activity = await withResolveContext({ backend }, () =>
      detectSpeechActivity(audio(), 20),
    );
    // Silence at 7.71-8.10 and 10.71-11.09 inverts to three activity spans,
    // and the two short gaps survive the default 0.1s minimum.
    expect(activity).toEqual([
      { start: 0, end: 7.70831 },
      { start: 8.10385, end: 10.7127 },
      { start: 11.0909, end: 20 },
    ]);
  });

  test("keeps the voice bandpass in the cloud filter chain", async () => {
    const { backend, calls } = makeCloudBackend(RENDI_METADATA, 20);
    await withResolveContext({ backend }, () =>
      detectSpeechActivity(audio(), 20, { band: { low: 300, high: 3000 } }),
    );

    const args = calls[0]!.outputArgs.join(" ");
    expect(args).toContain("highpass=f=300");
    expect(args).toContain("lowpass=f=3000");
    expect(args).toContain("ametadata=mode=print:file={{out_1}}");
  });

  test("does not probe — duration is already a parameter", async () => {
    let probes = 0;
    const { backend } = makeCloudBackend(RENDI_METADATA, 20);
    const spied = {
      ...backend,
      ffprobe: async () => {
        probes++;
        return { duration: 20, width: 0, height: 0 };
      },
    } as unknown as FFmpegBackend;

    await withResolveContext({ backend: spied }, () =>
      detectSpeechActivity(audio(), 20),
    );
    expect(probes).toBe(0);
  });
});

describe("local backend is untouched", () => {
  test("a backend named local still uses the stderr path", async () => {
    let ran = 0;
    const localish = {
      name: "local",
      async ffprobe() {
        return { duration: 2, width: 0, height: 0 };
      },
      async resolvePath() {
        return "/tmp/x.mp3";
      },
      async run() {
        ran++;
        return { output: { type: "file" as const, path: "/tmp/x.txt" } };
      },
    } as unknown as FFmpegBackend;

    // Real ffmpeg over real bytes; what matters is that backend.run() — the
    // cloud route — is never reached.
    const file = File.fromBuffer(new Uint8Array([0, 1, 2, 3]), "audio/mpeg");
    await withResolveContext({ backend: localish }, () =>
      detectSilence(file),
    ).catch(() => {});
    expect(ran).toBe(0);
    file.releaseTempFile();
  });
});

describe("no silent degradation inside a cloud render", () => {
  test("activity refinement runs instead of being swallowed", async () => {
    // AudioNode.transcribe() wraps the detectors in `try { … } catch { return raw }`.
    // While they threw on cloud backends, that catch turned a hard failure
    // into a quiet one: speechRange() returned raw whisper timings and the
    // caller had no way to tell refinement had been skipped. This asserts
    // the detector now succeeds, so the catch stays a real fallback rather
    // than the normal cloud path.
    const { backend, calls } = makeCloudBackend(RENDI_METADATA, 20);

    let threw = false;
    const activity = await withResolveContext({ backend }, async () => {
      try {
        return await detectSpeechActivity(audio(), 20);
      } catch {
        threw = true;
        return [];
      }
    });

    expect(threw).toBe(false);
    expect(activity.length).toBeGreaterThan(0);
    expect(calls.length).toBe(1);
  });
});
