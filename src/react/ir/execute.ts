/**
 * Plan executor — runs a CompiledPlan step by step.
 *
 * This is the generation phase of `render()`: every media-producing step
 * (image, video, speech, music, talking-head, audio extraction, slice,
 * ffmpeg, probe) is executed in dependency order with bounded parallelism,
 * and its result is written back onto the element as `meta`. After
 * execution the tree is fully materialized, so the compose phase
 * (renderRoot) finds every element pre-resolved and only assembles the
 * timeline.
 *
 * Dispatch reuses the existing renderers — they already short-circuit on
 * `element instanceof ResolvedElement` / `meta.file` and deduplicate via
 * `ctx.pendingFiles`, so executing a step is "render this element now and
 * record the file".
 */

import type { File } from "../../ai-sdk/file";
import { renderAudio } from "../renderers/audio";
import type { RenderContext } from "../renderers/context";
import { renderImage } from "../renderers/image";
import { renderMusic } from "../renderers/music";
import { renderSpeech } from "../renderers/speech";
import { renderTalkingHead } from "../renderers/talking-head";
import { renderVideo } from "../renderers/video";
import {
  resolveFFmpegElement,
  resolveProbeElement,
  resolveSliceElement,
} from "../resolve";
import { getResolveContext, withResolveContext } from "../resolve-context";
import type { VargElement } from "../types";
import type { CompiledPlan, Step, StepEvent } from "./types";

export interface ExecuteOptions {
  /** Max concurrent steps. Defaults to 3 (matches render concurrency). */
  concurrency?: number;
  /** Called for each step lifecycle event. */
  onEvent?: (event: StepEvent) => void;
  /**
   * Throw an aggregated error when steps fail (default true).
   *
   * `render()` passes `false`: failed steps leave their rejected promise in
   * `ctx.pendingFiles`, and the compose phase re-encounters the same
   * rejection with legacy per-clip error aggregation ("N of M clips
   * failed") — successful results are preserved and cached either way.
   */
  throwOnError?: boolean;
}

/** Probe duration for a media file so meta.duration is populated. */
async function probeStepDuration(
  file: File,
  ctx: RenderContext,
): Promise<number> {
  if (file.duration) return file.duration;
  try {
    const path = await ctx.backend.resolvePath(file);
    const info = await ctx.backend.ffprobe(path);
    return info.duration ?? 0;
  } catch {
    return 0;
  }
}

/**
 * Execute a single step by dispatching to the renderer for its element
 * type, then write the result into `element.meta`.
 */
async function runStep(step: Step, ctx: RenderContext): Promise<void> {
  const element = step.element;

  switch (step.kind) {
    case "generate-image": {
      const file = await renderImage(element as VargElement<"image">, ctx);
      element.meta = { file, duration: 0 };
      return;
    }
    case "generate-video": {
      const file = await renderVideo(element as VargElement<"video">, ctx);
      element.meta = {
        file,
        duration: await probeStepDuration(file, ctx),
      };
      return;
    }
    case "generate-speech": {
      const file = await renderSpeech(element as VargElement<"speech">, ctx);
      element.meta = {
        file,
        duration: await probeStepDuration(file, ctx),
      };
      return;
    }
    case "generate-music": {
      const file = await renderMusic(element as VargElement<"music">, ctx);
      element.meta = {
        file,
        duration: await probeStepDuration(file, ctx),
      };
      return;
    }
    case "talking-head": {
      const file = await renderTalkingHead(
        element as VargElement<"talking-head">,
        ctx,
      );
      element.meta = {
        file,
        duration: await probeStepDuration(file, ctx),
      };
      return;
    }
    case "extract-audio": {
      const file = await renderAudio(element as VargElement<"audio">, ctx);
      element.meta = {
        file,
        duration: await probeStepDuration(file, ctx),
        // Inherit word timings from a resolved speech parent, if any.
        words: (element.props.parent as VargElement | undefined)?.meta?.words,
      };
      return;
    }
    case "slice": {
      const resolved = await resolveSliceElement(
        element as VargElement<"slice">,
        element.props as unknown as Parameters<typeof resolveSliceElement>[1],
      );
      element.meta = resolved.meta;
      return;
    }
    case "ffmpeg": {
      const resolved = await resolveFFmpegElement(
        element as VargElement<"ffmpeg">,
        element.props as unknown as Parameters<typeof resolveFFmpegElement>[1],
      );
      element.meta = resolved.meta;
      return;
    }
    case "probe": {
      const resolved = await resolveProbeElement(
        element as VargElement<"probe">,
        element.props as unknown as Parameters<typeof resolveProbeElement>[1],
      );
      element.meta = resolved.meta;
      return;
    }
    case "transcribe":
    case "silence-detect":
      // Analysis ops are pulled on demand via AudioNode methods; they are
      // not materialized as part of the render plan (no meta to produce).
      return;
    case "compose":
      // The compose step is executed by renderRoot after the plan finishes.
      return;
  }
}

/**
 * Execute all steps of a plan in dependency order with bounded parallelism.
 *
 * A step starts as soon as all of its dependencies are done (not in rigid
 * topological batches), capped at `concurrency` simultaneous steps.
 * Failures propagate after all in-flight steps settle; the first error is
 * rethrown (matching previous render semantics), remaining pending steps
 * are not started.
 */
export async function executePlan(
  plan: CompiledPlan,
  ctx: RenderContext,
  options: ExecuteOptions = {},
): Promise<void> {
  const concurrency = options.concurrency ?? 3;
  const onEvent = options.onEvent;

  const executable = plan.steps.filter((s) => s.kind !== "compose");
  onEvent?.({ type: "plan", totalSteps: executable.length });

  const done = new Set<Step>();
  const failed: Error[] = [];
  const inFlight = new Map<Step, Promise<void>>();
  const pending = new Set<Step>(executable);

  // Pre-resolved steps are immediately done.
  for (const step of executable) {
    if (step.status === "skipped" || step.element.meta?.file) {
      pending.delete(step);
      done.add(step);
      if (step.status !== "skipped") step.status = "done";
      onEvent?.({ type: "step-skipped", step });
    }
  }

  const isReady = (step: Step) =>
    step.dependsOn.every((dep) => dep.kind === "compose" || done.has(dep));

  // Capture the ambient resolve context (set by render()) so steps that
  // run in later event-loop turns still see backend/cache/storage.
  const resolveCtx = getResolveContext();

  const startStep = (step: Step) => {
    pending.delete(step);
    step.status = "running";
    onEvent?.({ type: "step-start", step });

    const run = () => runStep(step, ctx);
    const promise = (resolveCtx ? withResolveContext(resolveCtx, run) : run())
      .then(() => {
        step.status = "done";
        done.add(step);
        onEvent?.({ type: "step-complete", step });
      })
      .catch((err) => {
        step.status = "failed";
        const error = err instanceof Error ? err : new Error(String(err));
        failed.push(error);
        onEvent?.({ type: "step-failed", step, error });
      })
      .finally(() => {
        inFlight.delete(step);
      });
    inFlight.set(step, promise);
  };

  while (pending.size > 0 || inFlight.size > 0) {
    // Launch every ready step up to the concurrency cap (unless failing).
    if (failed.length === 0) {
      for (const step of [...pending]) {
        if (inFlight.size >= concurrency) break;
        if (isReady(step)) startStep(step);
      }
    }

    if (inFlight.size === 0) {
      if (failed.length > 0) break;
      if (pending.size > 0) {
        // No in-flight work and nothing ready — unsatisfiable deps.
        const stuck = [...pending].map((s) => s.id).join(", ");
        throw new Error(
          `executePlan deadlock: steps [${stuck}] have unsatisfiable dependencies`,
        );
      }
      break;
    }

    // Wait for any in-flight step to settle, then re-evaluate readiness.
    await Promise.race(inFlight.values());
  }

  if (failed.length > 0 && (options.throwOnError ?? true)) {
    if (failed.length === 1) throw failed[0];
    throw new Error(
      `${failed.length} plan steps failed: ${failed.map((e) => e.message).join("; ")}`,
    );
  }
}
