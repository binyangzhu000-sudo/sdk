/**
 * AudioNode — the derived audio element returned by `video.audio` and
 * `speech.audio`.
 *
 * A lazy, memoized VargElement<"audio"> that:
 * - resolves on `await` (video parent → ffmpeg -vn extraction; speech
 *   parent → reuses the speech audio file + word timings)
 * - exposes analysis helpers: `transcribe()`, `silenceSegments()`, `range()`,
 *   `speechRange()`
 * - is accepted everywhere a speech element is accepted (clip child,
 *   `prompt.audio`, TalkingHead audio, `Captions src`)
 *
 * Unlike the factory elements (Image/Video/Speech), the `.then` here is
 * memoized instead of self-deleting: re-awaiting returns the same
 * ResolvedElement, and `resolveLazy` strips `.then` from tree copies so
 * `Promise.all` never triggers accidental resolution.
 */

import type { WordTiming } from "../speech/types";
import type {
  SilenceDetectOptions,
  TimeRange,
  TranscriptionResult,
} from "./primitives/audio";
import {
  computeSoundBounds,
  detectSilence,
  transcribeAudio,
} from "./primitives/audio";
import { getActiveCache, resolveAudioElement } from "./resolve";
import type { ResolvedElement } from "./resolved-element";
import type { AudioElementProps, VargElement } from "./types";

export interface AudioNode
  extends VargElement<"audio">,
    PromiseLike<ResolvedElement<"audio">> {
  /**
   * Transcribe the audio to `{ text, words }` with word-level timestamps.
   * When the parent speech element carries native ElevenLabs word timings,
   * they are reused and no transcription call is made. Cached by content hash.
   */
  transcribe(): Promise<TranscriptionResult>;
  /**
   * Detect silence intervals via ffmpeg `silencedetect`.
   * Note: detects *sound*, not speech — ambient noise counts as sound.
   */
  silenceSegments(options?: SilenceDetectOptions): Promise<TimeRange[]>;
  /**
   * Range of audible content: start of first sound, end of last sound.
   * Derived from `silenceSegments()`.
   *
   * Note: this is *sound*, not speech — ambient noise, footsteps, and music
   * count. For "where do they actually talk", use `speechRange()`.
   */
  range(options?: SilenceDetectOptions): Promise<TimeRange>;
  /**
   * Range of actual speech: start of the first spoken word, end of the
   * last one, from word-level transcription timings (`transcribe()` —
   * native ElevenLabs alignment when present, whisper otherwise; cached).
   *
   * Returns `null` when no speech is detected (empty transcript). Word
   * boundaries from whisper carry ~±0.1s slack — pass `pad` to widen the
   * range for soft trims (clamped to [0, duration]).
   */
  speechRange(options?: { pad?: number }): Promise<TimeRange | null>;
  /**
   * Duration in seconds. Available synchronously when derived from an
   * already-resolved parent (e.g. `const { audio } = await Speech(...)`);
   * 0 before resolution otherwise.
   */
  readonly duration: number;
  /** Word-level timings inherited from a resolved speech parent, if any. */
  readonly words: WordTiming[] | undefined;
}

/** Parent types whose file is already an audio track (no extraction needed). */
const AUDIO_FILE_PARENTS = new Set(["speech", "audio", "music"]);

/**
 * Create an AudioNode derived from a parent element (or a direct src).
 *
 * When the parent is an already-resolved speech element, the node is born
 * pre-resolved (meta set from the parent's file + words) — renderers and
 * compile() treat it as done.
 */
export function makeAudioNode(props: AudioElementProps): AudioNode {
  const element: VargElement<"audio"> = {
    type: "audio",
    props: props as Record<string, unknown>,
    children: [],
  };

  const parent = props.parent;
  // Pre-seed meta when the parent already carries an *audio* file.
  // Video parents still require extraction even when resolved.
  if (parent?.meta?.file && AUDIO_FILE_PARENTS.has(parent.type)) {
    element.meta = {
      file: parent.meta.file,
      duration: parent.meta.duration,
      words: parent.meta.words,
    };
  }

  const node = element as AudioNode;
  let resolvedPromise: Promise<ResolvedElement<"audio">> | undefined;
  const resolveOnce = () => {
    // Sync element.meta after resolution so the sync getters
    // (.duration, .words) and renderers see the resolved file.
    resolvedPromise ??= resolveAudioElement(element).then((r) => {
      element.meta = r.meta;
      return r;
    });
    return resolvedPromise;
  };

  // Memoized thenable — safe to re-await (unlike makeThenable's
  // consume-on-first-call semantics).
  // biome-ignore lint/suspicious/noThenProperty: intentional — makes element awaitable
  node.then = (onFulfilled, onRejected) =>
    resolveOnce().then(onFulfilled, onRejected);

  node.transcribe = async (): Promise<TranscriptionResult> => {
    const resolved = await resolveOnce();
    const words = resolved.meta.words;
    if (words && words.length > 0) {
      return { text: joinWords(words), words };
    }
    return transcribeAudio(resolved.meta.file, { cache: getActiveCache() });
  };

  node.silenceSegments = async (
    options?: SilenceDetectOptions,
  ): Promise<TimeRange[]> => {
    const resolved = await resolveOnce();
    return detectSilence(resolved.meta.file, options);
  };

  node.range = async (options?: SilenceDetectOptions): Promise<TimeRange> => {
    const resolved = await resolveOnce();
    const silences = await detectSilence(resolved.meta.file, options);
    return computeSoundBounds(silences, resolved.meta.duration);
  };

  node.speechRange = async (options?: {
    pad?: number;
  }): Promise<TimeRange | null> => {
    const resolved = await resolveOnce();
    const { words } = await node.transcribe();
    if (!words || words.length === 0) return null;

    const pad = options?.pad ?? 0;
    const first = words[0]!;
    const last = words[words.length - 1]!;
    const duration = resolved.meta.duration;
    return {
      start: Math.max(0, first.start - pad),
      end: duration > 0 ? Math.min(duration, last.end + pad) : last.end + pad,
    };
  };

  // Sync convenience getters — populated when meta is pre-seeded (resolved
  // parent) or after the node has been awaited.
  Object.defineProperties(node, {
    duration: {
      get() {
        return element.meta?.duration ?? 0;
      },
      enumerable: false,
    },
    words: {
      get() {
        return element.meta?.words;
      },
      enumerable: false,
    },
  });

  return node;
}

function joinWords(words: WordTiming[]): string {
  return words.map((w) => w.word).join(" ");
}
