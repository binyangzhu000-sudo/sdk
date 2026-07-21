import { computeCacheKey } from "../renderers/utils";
import { ResolvedElement } from "../resolved-element";
import type { VargElement, VargNode } from "../types";
import { topoSort } from "./topo-sort";
import { CompiledPlan, type Diagnostic, type OpKind, type Step } from "./types";

/** Element types that map 1:1 to a generation/processing Step. */
const OP_BY_TYPE: Partial<Record<VargElement["type"], OpKind>> = {
  image: "generate-image",
  video: "generate-video",
  speech: "generate-speech",
  music: "generate-music",
  "talking-head": "talking-head",
  audio: "extract-audio",
  slice: "slice",
  ffmpeg: "ffmpeg",
  probe: "probe",
};

/** Container types we recurse into via children. */
const CONTAINER_TYPES = new Set([
  "render",
  "clip",
  "overlay",
  "split",
  "slider",
  "swipe",
]);

function isVargElement(v: unknown): v is VargElement {
  return (
    typeof v === "object" &&
    v !== null &&
    "type" in v &&
    "props" in v &&
    "children" in v
  );
}

function isPreResolved(element: VargElement): boolean {
  return element instanceof ResolvedElement || element.meta?.file != null;
}

function truncate(s: string, n = 30): string {
  return s.length > n ? `${s.slice(0, n)}...` : s;
}

function getLabel(element: VargElement): string {
  const props = element.props;
  const prompt = props.prompt;
  if (typeof prompt === "string") return `${element.type}: ${truncate(prompt)}`;
  if (prompt && typeof prompt === "object" && "text" in prompt) {
    const text = (prompt as { text?: string }).text;
    if (text) return `${element.type}: ${truncate(text)}`;
  }
  if (typeof props.src === "string")
    return `${element.type}: ${String(props.src).split("/").pop()}`;
  const text = element.children
    .filter((c): c is string => typeof c === "string")
    .join(" ");
  if (text) return `${element.type}: ${truncate(text)}`;
  return element.type;
}

/**
 * Compile a VargElement tree into an ordered, validated execution plan.
 *
 * Read-only: never triggers generation. Pre-resolved elements (from
 * `await Video(...)` etc.) become Steps with status "skipped".
 *
 * ```tsx
 * const plan = compile(tree);
 * if (plan.hasErrors()) throw new Error(plan.errors[0].message);
 * for (const step of plan.steps) console.log(step.kind, step.label);
 * ```
 */
export function compile(root: VargElement): CompiledPlan {
  const diagnostics: Diagnostic[] = [];
  const steps: Step[] = [];
  /** Dedup: identical elements (by reference) map to the same step. */
  const stepByElement = new Map<VargElement, Step>();
  let counter = 0;

  if (root.type !== "render") {
    diagnostics.push({
      severity: "error",
      code: "VARG_ROOT_NOT_RENDER",
      message: `Root element must be <Render>, got <${root.type}>`,
      path: [root.type],
    });
  }

  function addDiagnostic(
    severity: Diagnostic["severity"],
    code: string,
    message: string,
    path: string[],
    stepId?: string,
  ) {
    diagnostics.push({ severity, code, message, path, stepId });
  }

  function validateGenerable(element: VargElement, path: string[]) {
    const props = element.props;
    // Speech generates from text children; audio derives from parent/src;
    // talking-head from image+audio props (validated at render time).
    const hasTextChildren = element.children.some(
      (c) => typeof c === "string" && c.length > 0,
    );
    const hasSource =
      props.src != null ||
      props.prompt != null ||
      isPreResolved(element) ||
      (element.type === "speech" && hasTextChildren) ||
      (element.type === "audio" && props.parent != null) ||
      element.type === "talking-head";

    if (!hasSource) {
      const requirement =
        element.type === "speech"
          ? "text children"
          : element.type === "audio"
            ? "'parent' or 'src'"
            : "'prompt' or 'src'";
      addDiagnostic(
        "error",
        "VARG_MISSING_SOURCE",
        `<${element.type}> requires ${requirement}`,
        path,
      );
    }
    // Model required when generating (prompt without src), except audio/slice/ffmpeg/probe
    if (
      props.prompt != null &&
      props.src == null &&
      props.model == null &&
      !isPreResolved(element) &&
      ["image", "video", "speech", "music", "talking-head"].includes(
        element.type,
      )
    ) {
      addDiagnostic(
        "warning",
        "VARG_MISSING_MODEL",
        `<${element.type}> has a prompt but no 'model' — will fall back to defaults at render time`,
        path,
      );
    }

    if (element.type === "video") {
      validateVideoAudioProps(element, path);
    }

    if (
      element.type !== "video" &&
      element.type !== "audio" &&
      props.keepAudio !== undefined &&
      element.type !== "talking-head"
    ) {
      addDiagnostic(
        "warning",
        "VARG_KEEP_AUDIO_IGNORED",
        `'keepAudio' has no effect on <${element.type}>`,
        path,
      );
    }
  }

  function validateVideoAudioProps(element: VargElement, path: string[]) {
    const props = element.props;
    if (props.audio !== "native") return;

    if (props.keepAudio === false) {
      addDiagnostic(
        "error",
        "VARG_AUDIO_CONFLICT",
        `audio: "native" conflicts with keepAudio: false — native audio would be generated and then dropped. Remove one of them.`,
        path,
      );
    }
    const providerOptions = props.providerOptions as
      | Record<string, Record<string, unknown>>
      | undefined;
    if (providerOptions) {
      for (const [providerKey, opts] of Object.entries(providerOptions)) {
        if (opts && opts.generate_audio === false) {
          addDiagnostic(
            "error",
            "VARG_AUDIO_CONFLICT",
            `audio: "native" conflicts with providerOptions.${providerKey}.generate_audio: false`,
            path,
          );
        }
      }
    }
  }

  /**
   * Register a Step for a generable element and return it.
   * Recursively discovers dependencies in nested prompt props / children.
   */
  function addStep(element: VargElement, path: string[]): Step {
    const existing = stepByElement.get(element);
    if (existing) return existing;

    const kind = OP_BY_TYPE[element.type];
    if (!kind) {
      throw new Error(`No op kind for element type "${element.type}"`);
    }

    validateGenerable(element, path);

    const step: Step = {
      id: `step-${counter++}`,
      kind,
      element,
      dependsOn: [],
      cacheKey: computeCacheKey(element).join(":"),
      status: isPreResolved(element) ? "skipped" : "pending",
      label: getLabel(element),
      path,
    };
    stepByElement.set(element, step);
    steps.push(step);

    // Discover dependencies in prompt props
    const prompt = element.props.prompt;
    if (prompt && typeof prompt === "object") {
      const p = prompt as Record<string, unknown>;
      if (Array.isArray(p.images)) {
        for (const img of p.images) {
          if (isVargElement(img)) {
            step.dependsOn.push(addStep(img, [...path, "prompt.images"]));
          }
        }
      }
      for (const key of ["audio", "video"] as const) {
        const v = p[key];
        if (isVargElement(v)) {
          step.dependsOn.push(addStep(v, [...path, `prompt.${key}`]));
        }
      }
    }

    // TalkingHead: image / audio props
    for (const key of ["image", "audio"] as const) {
      const v = element.props[key];
      if (isVargElement(v)) {
        step.dependsOn.push(addStep(v, [...path, key]));
      }
    }

    // Slice / FFmpeg / Probe / Audio: src prop may be an element
    const src = element.props.src;
    if (isVargElement(src)) {
      step.dependsOn.push(addStep(src, [...path, "src"]));
    }
    // Audio element: parent video reference
    const parent = element.props.parent;
    if (isVargElement(parent)) {
      step.dependsOn.push(addStep(parent, [...path, "parent"]));
    }

    return step;
  }

  const composeDeps: Step[] = [];

  function walk(node: VargNode, path: string[], clipIndex?: number) {
    if (node == null || typeof node === "string" || typeof node === "number")
      return;
    if (Array.isArray(node)) {
      for (const child of node) walk(child, path, clipIndex);
      return;
    }
    if (!isVargElement(node)) return;

    const element = node;
    const label =
      element.type === "clip" && clipIndex !== undefined
        ? `clip[${clipIndex}]`
        : element.type;
    const currentPath = [...path, label];

    if (OP_BY_TYPE[element.type]) {
      composeDeps.push(addStep(element, currentPath));
      return;
    }

    if (element.type === "captions") {
      const src = element.props.src;
      if (isVargElement(src)) {
        composeDeps.push(addStep(src, [...currentPath, "src"]));
      }
      return;
    }

    if (element.type === "packshot") {
      for (const key of ["background", "logo"] as const) {
        const v = element.props[key];
        if (isVargElement(v)) {
          composeDeps.push(addStep(v, [...currentPath, key]));
        }
      }
      return;
    }

    if (element.type === "__lazy") {
      addDiagnostic(
        "warning",
        "VARG_UNRESOLVED_LAZY",
        "Tree contains an unresolved async component — call resolveLazy() before compile() for a complete plan",
        currentPath,
      );
      return;
    }

    if (CONTAINER_TYPES.has(element.type) || element.type === "clip") {
      // Missing-duration warning for clips
      if (element.type === "clip") {
        const d = element.props.duration;
        if (d === undefined || d === "auto") {
          const hasMedia = element.children.some(
            (c) =>
              isVargElement(c) && (c.type === "video" || c.type === "image"),
          );
          if (!hasMedia || d === undefined) {
            addDiagnostic(
              "info",
              "VARG_DURATION_DEFAULT",
              `<Clip> without a numeric duration falls back to 3s at render time`,
              currentPath,
            );
          }
        }
      }
      let childClipIndex = 0;
      for (const child of element.children) {
        const idx =
          isVargElement(child) && child.type === "clip"
            ? childClipIndex++
            : undefined;
        walk(child, currentPath, idx);
      }
    }
  }

  let clipIdx = 0;
  if (root.type === "render") {
    for (const child of root.children) {
      const idx =
        isVargElement(child) && child.type === "clip" ? clipIdx++ : undefined;
      walk(child, ["render"], idx);
    }
  }

  // Final compose step (the editly composition) depends on everything.
  if (root.type === "render") {
    const composeStep: Step = {
      id: `step-${counter++}`,
      kind: "compose",
      element: root,
      dependsOn: composeDeps,
      cacheKey: "",
      status: "pending",
      label: "compose",
      path: ["render"],
    };
    steps.push(composeStep);
  }

  let ordered: Step[];
  try {
    ordered = topoSort(steps);
  } catch (err) {
    addDiagnostic(
      "error",
      "VARG_CIRCULAR_DEPENDENCY",
      err instanceof Error ? err.message : String(err),
      ["render"],
    );
    ordered = steps;
  }

  return new CompiledPlan(ordered, diagnostics);
}
