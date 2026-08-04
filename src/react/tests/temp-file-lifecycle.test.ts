/**
 * End-to-end temp-file accounting for the buffer-backed audio path.
 *
 * `extractAudio` returns `File.fromBuffer(bytes, "audio/mpeg")` — no URL — so
 * every downstream consumer falls through `file.url ?? await file.toTempFile()`
 * to the materialization branch. For one clip of `speechRange()` that is three
 * consumers: the duration probe, `detectSilence`, and `detectSpeechActivity`.
 *
 * Two coupled defects lived here. `toTempFile()` minted a fresh path per call,
 * so one clip's audio was written to disk three times; and cleanup belonged to
 * each caller, applied inconsistently — silence/speech-activity unlinked in
 * `finally`, the probe never did and leaked one file per clip.
 *
 * Note that *surviving* file counts cannot tell the two versions apart: the old
 * code leaked one file from the probe and deleted two, the new code keeps one
 * owned file — both leave one behind. The discriminating signals are how many
 * times bytes were WRITTEN, and whether a shared path is still readable after
 * the first consumer returns. These tests assert those directly, over real
 * ffmpeg runs against a real tone.
 */

import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  test,
} from "bun:test";
import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { $ } from "bun";
import { File } from "../../ai-sdk/file";
import { detectSilence } from "../primitives/silence";
import { detectSpeechActivity } from "../primitives/speech-activity";

/**
 * Record every path `toTempFile()` hands out for one File. Distinct paths mean
 * distinct writes of the same bytes.
 */
function spyTempPaths(file: File): string[] {
  const paths: string[] = [];
  const original = file.toTempFile.bind(file);
  (file as unknown as { toTempFile: () => Promise<string> }).toTempFile =
    async () => {
      const path = await original();
      paths.push(path);
      return path;
    };
  return paths;
}

/**
 * Bytes of a real 2s mp3: 0.5s silence, 1s 440Hz tone, 0.5s silence — leading
 * and trailing silence so both detectors have something to find.
 */
let audioBytes: Uint8Array;
let fixturePath: string;

/** Per-test TMPDIR, so temp files are counted in isolation from the real /tmp. */
let sandbox: string;
let originalTmpdir: string | undefined;

beforeAll(async () => {
  fixturePath = join(tmpdir(), `tempfile-fixture-${Date.now()}.mp3`);
  // The filter goes through a variable: a literal `|` in the template would be
  // parsed by Bun's shell as a pipe. `-t 2` bounds the output — apad alone
  // pads forever.
  const filter = "adelay=500|500,apad=pad_dur=0.5";
  const result =
    await $`ffmpeg -y -f lavfi -i sine=frequency=440:duration=1 -af ${filter} -t 2 -acodec libmp3lame ${fixturePath}`
      .quiet()
      .nothrow();
  if (result.exitCode !== 0) {
    throw new Error(`fixture generation failed: ${result.stderr.toString()}`);
  }
  audioBytes = new Uint8Array(await Bun.file(fixturePath).arrayBuffer());

  originalTmpdir = process.env.TMPDIR;
});

afterAll(async () => {
  await Bun.file(fixturePath).delete?.();
  if (originalTmpdir === undefined) delete process.env.TMPDIR;
  else process.env.TMPDIR = originalTmpdir;
});

async function withSandbox<T>(fn: () => Promise<T>): Promise<T> {
  sandbox = await mkdtemp(join(tmpdir(), "varg-temp-sandbox-"));
  process.env.TMPDIR = sandbox;
  try {
    return await fn();
  } finally {
    process.env.TMPDIR = originalTmpdir;
  }
}

afterEach(async () => {
  if (sandbox) await rm(sandbox, { recursive: true, force: true });
});

/** Local ffprobe duration — mirrors resolve.ts's probeDurationLocal. */
async function probeDuration(file: File): Promise<number> {
  const target = file.url ?? (await file.toTempFile());
  const result =
    await $`ffprobe -v error -show_entries format=duration -of json ${target}`.json();
  const duration = Number.parseFloat(result?.format?.duration ?? "0");
  return Number.isFinite(duration) ? duration : 0;
}

describe("buffer-backed audio: one materialization per File", () => {
  test("three consumers of one clip write the file once", async () => {
    await withSandbox(async () => {
      // Exactly what extractAudio hands back: buffer-backed, no URL.
      const audio = File.fromBuffer(audioBytes, "audio/mpeg");
      const handedOut = spyTempPaths(audio);

      const duration = await probeDuration(audio);
      await detectSilence(audio);
      await detectSpeechActivity(audio, duration);

      // Old behavior: three calls, three distinct `varg-<ts>-<rand>` paths.
      expect(handedOut.length).toBe(3);
      expect(new Set(handedOut).size).toBe(1);

      // And only one file was ever created in the sandbox.
      expect((await readdir(sandbox)).length).toBe(1);
    });
  });

  test("the shared path survives each consumer", async () => {
    await withSandbox(async () => {
      const audio = File.fromBuffer(audioBytes, "audio/mpeg");
      const path = await audio.toTempFile();

      // detectSilence used to unlink this in `finally` — the next consumer got
      // handed a path to nothing.
      await detectSilence(audio);
      expect(await Bun.file(path).exists()).toBe(true);

      await detectSpeechActivity(audio, 2);
      expect(await Bun.file(path).exists()).toBe(true);
    });
  });

  test("later consumers still read real audio, not an emptied path", async () => {
    await withSandbox(async () => {
      const audio = File.fromBuffer(audioBytes, "audio/mpeg");

      const duration = await probeDuration(audio);
      const silences = await detectSilence(audio);
      const activity = await detectSpeechActivity(audio, duration);

      // The fixture is 0.5s silence / 1s tone / 0.5s silence: every stage must
      // see that structure, including the ones that ran last.
      expect(duration).toBeGreaterThan(1.9);
      expect(silences.length).toBe(2);
      expect(activity.length).toBe(1);
      expect(activity[0]!.start).toBeGreaterThan(0.4);
      expect(activity[0]!.end).toBeLessThan(1.6);
    });
  });

  test("12 clips write 12 files, not 24-36", async () => {
    await withSandbox(async () => {
      // The ep5 shape: one File per clip, three consumers each.
      const clips = Array.from({ length: 12 }, () =>
        File.fromBuffer(audioBytes, "audio/mpeg"),
      );
      const spies = clips.map(spyTempPaths);

      await Promise.all(
        clips.map(async (clip) => {
          const duration = await probeDuration(clip);
          await detectSilence(clip);
          await detectSpeechActivity(clip, duration);
        }),
      );

      expect(spies.flat().length).toBe(36);
      expect(new Set(spies.flat()).size).toBe(12);
      expect((await readdir(sandbox)).length).toBe(12);
    });
  }, 30_000);

  test("releaseTempFile leaves nothing behind", async () => {
    await withSandbox(async () => {
      const audio = File.fromBuffer(audioBytes, "audio/mpeg");

      const duration = await probeDuration(audio);
      await detectSilence(audio);
      await detectSpeechActivity(audio, duration);
      audio.releaseTempFile();

      // The probe used to leak its copy unconditionally — no finally, no delete.
      expect(await readdir(sandbox)).toEqual([]);
    });
  });
});

describe("url-backed audio", () => {
  test("never materializes — ffmpeg reads the URL directly", async () => {
    await withSandbox(async () => {
      const audio = File.fromUrl("https://example.invalid/a.mp3", "audio/mpeg");

      // ffmpeg fails to fetch; what matters is that nothing touched TMPDIR.
      await detectSilence(audio).catch(() => {});

      expect(await readdir(sandbox)).toEqual([]);
    });
  });
});
