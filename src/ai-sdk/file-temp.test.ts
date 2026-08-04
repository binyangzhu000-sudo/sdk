/**
 * `File.toTempFile()` materialization ownership.
 *
 * A buffer-backed File (the return of `extractAudio`, which has no URL) is
 * consumed by three separate ffmpeg passes for one clip: `probeDurationLocal`,
 * `detectSilence`, and `detectSpeechActivity`. Each of them needs a *path*,
 * so each called `toTempFile()`.
 *
 * Two coupled bugs came out of that:
 *
 *  1. `toTempFile()` minted a fresh `varg-<ts>-<rand>` path on every call, so
 *     the same bytes were written to disk 2-3x per clip (12 clips of ep5 →
 *     ~36 writes of identical audio).
 *  2. Cleanup was the caller's job, applied inconsistently — silence and
 *     speech-activity deleted in `finally`, `probeDurationLocal` never did and
 *     leaked one file per clip.
 *
 * Memoizing (1) without moving ownership would have broken (2) the other way:
 * the second consumer would receive a path the first consumer had deleted. So
 * the File owns the path, and these tests pin both halves — same path across
 * calls, and the file still readable after every consumer is done.
 */

import { afterEach, describe, expect, test } from "bun:test";
import { readdir, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { File } from "./file";

const created: File[] = [];

function track(file: File): File {
  created.push(file);
  return file;
}

afterEach(() => {
  for (const f of created) f.releaseTempFile();
  created.length = 0;
});

function tempDir(): string {
  return process.env.TMPDIR ?? tmpdir();
}

async function countVargTemps(): Promise<number> {
  const entries = await readdir(tempDir());
  return entries.filter((e) => e.startsWith("varg-")).length;
}

describe("File.toTempFile memoization", () => {
  test("returns the same path across repeated calls", async () => {
    const file = track(
      File.fromBuffer(new Uint8Array([1, 2, 3]), "audio/mpeg"),
    );

    const first = await file.toTempFile();
    const second = await file.toTempFile();
    const third = await file.toTempFile();

    expect(second).toBe(first);
    expect(third).toBe(first);
  });

  test("writes exactly one temp file for N consumers", async () => {
    const before = await countVargTemps();
    const file = track(
      File.fromBuffer(new Uint8Array([1, 2, 3]), "audio/mpeg"),
    );

    // The ep5 shape: probe, then silencedetect, then bandpassed silencedetect.
    await file.toTempFile();
    await file.toTempFile();
    await file.toTempFile();

    expect((await countVargTemps()) - before).toBe(1);
  });

  test("concurrent calls share one materialization", async () => {
    const before = await countVargTemps();
    const file = track(
      File.fromBuffer(new Uint8Array([9, 9, 9]), "audio/mpeg"),
    );

    // Consumers run under pMap concurrency — they can race into toTempFile()
    // before any of them has written, so memoizing on the settled path alone
    // is not enough; the in-flight promise must be shared too.
    const paths = await Promise.all([
      file.toTempFile(),
      file.toTempFile(),
      file.toTempFile(),
      file.toTempFile(),
    ]);

    expect(new Set(paths).size).toBe(1);
    expect((await countVargTemps()) - before).toBe(1);
  });

  test("materialized file holds the file's bytes and the right extension", async () => {
    const bytes = new Uint8Array([0xff, 0xfb, 0x90, 0x00]);
    const file = track(File.fromBuffer(bytes, "audio/mpeg"));

    const path = await file.toTempFile();

    expect(path.endsWith(".mp3")).toBe(true);
    expect(new Uint8Array(await Bun.file(path).arrayBuffer())).toEqual(bytes);
  });

  test("path stays readable after every consumer has finished", async () => {
    const file = track(
      File.fromBuffer(new Uint8Array([4, 5, 6]), "audio/mpeg"),
    );

    // Previously detectSilence deleted this in `finally`, so the next
    // consumer's ffmpeg got ENOENT on a path it had just been handed.
    const consumerA = await file.toTempFile();
    const consumerB = await file.toTempFile();

    expect(await Bun.file(consumerA).exists()).toBe(true);
    expect(await Bun.file(consumerB).exists()).toBe(true);
  });

  test("re-materializes when the file is deleted externally", async () => {
    const file = track(File.fromBuffer(new Uint8Array([7, 7]), "audio/mpeg"));

    const first = await file.toTempFile();
    await Bun.file(first).delete?.();

    const second = await file.toTempFile();

    expect(await Bun.file(second).exists()).toBe(true);
    expect(new Uint8Array(await Bun.file(second).arrayBuffer())).toEqual(
      new Uint8Array([7, 7]),
    );
  });

  test("distinct Files get distinct temp paths", async () => {
    const a = track(File.fromBuffer(new Uint8Array([1]), "audio/mpeg"));
    const b = track(File.fromBuffer(new Uint8Array([2]), "audio/mpeg"));

    expect(await a.toTempFile()).not.toBe(await b.toTempFile());
  });

  test("loader-backed files read the source only once", async () => {
    // `File.fromPath` is loader-backed: repeated materialization would mean
    // repeated disk reads of the source video, not just repeated writes.
    const srcPath = `${tempDir()}/varg-test-src-${Date.now()}.mp3`;
    await Bun.write(srcPath, new Uint8Array([1, 2]));
    try {
      const lazy = track(File.fromPath(srcPath));

      const first = await lazy.toTempFile();
      const second = await lazy.toTempFile();

      expect(second).toBe(first);
      expect(new Uint8Array(await Bun.file(first).arrayBuffer())).toEqual(
        new Uint8Array([1, 2]),
      );
    } finally {
      await Bun.file(srcPath).delete?.();
    }
  });
});

describe("File.releaseTempFile", () => {
  test("deletes the materialized file", async () => {
    const file = File.fromBuffer(new Uint8Array([1, 2]), "audio/mpeg");
    const path = await file.toTempFile();
    expect(await Bun.file(path).exists()).toBe(true);

    file.releaseTempFile();

    expect(await Bun.file(path).exists()).toBe(false);
  });

  test("is a no-op when nothing was materialized", () => {
    const file = File.fromBuffer(new Uint8Array([1]), "audio/mpeg");
    expect(() => file.releaseTempFile()).not.toThrow();
  });

  test("re-materializes on demand after release", async () => {
    const file = track(File.fromBuffer(new Uint8Array([3]), "audio/mpeg"));

    const first = await file.toTempFile();
    file.releaseTempFile();
    const second = await file.toTempFile();

    expect(second).not.toBe(first);
    expect(await Bun.file(second).exists()).toBe(true);
  });

  test("leaves no temp file behind across a full consumer cycle", async () => {
    const before = await countVargTemps();

    const file = File.fromBuffer(new Uint8Array([1, 2, 3]), "audio/mpeg");
    await file.toTempFile();
    await file.toTempFile();
    file.releaseTempFile();

    expect(await countVargTemps()).toBe(before);
  });
});

describe("URL-backed files", () => {
  test("consumers short-circuit on url and never materialize", async () => {
    const before = await countVargTemps();
    const file = File.fromUrl("https://example.com/a.mp3");

    // Mirrors the `file.url ?? (await file.toTempFile())` guard in every
    // consumer: a URL-backed File is streamed by ffmpeg over HTTP range
    // requests, so it must not touch /tmp at all.
    const input = file.url ?? (await file.toTempFile());

    expect(input).toBe("https://example.com/a.mp3");
    expect(await countVargTemps()).toBe(before);
  });
});

describe("temp path hygiene", () => {
  test("path lives under TMPDIR", async () => {
    const file = track(File.fromBuffer(new Uint8Array([1]), "audio/mpeg"));
    const path = await file.toTempFile();

    expect(path.startsWith(tempDir())).toBe(true);
    expect(await stat(path)).toBeTruthy();
  });
});
