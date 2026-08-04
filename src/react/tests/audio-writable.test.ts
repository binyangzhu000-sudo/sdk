import { describe, expect, test } from "bun:test";
import { Speech, Video } from "../elements";

describe(".audio is writable (render dry-run regression)", () => {
  test("assigning to .audio does not throw", () => {
    const el = Speech({ voice: "adam", children: "hi" }) as any;
    expect(() => {
      el.audio = el;
    }).not.toThrow();
    expect(el.audio).toBe(el);
  });
  test("Video().audio assignable too", () => {
    const el = Video({ prompt: "x" }) as any;
    expect(() => {
      el.audio = { fake: true };
    }).not.toThrow();
    expect(el.audio.fake).toBe(true);
  });
  test("lazy getter still memoized when untouched", () => {
    const el = Video({ prompt: "x" });
    expect(el.audio).toBe(el.audio);
    expect(typeof el.audio.speechRange).toBe("function");
  });
});
