import { File } from "../../ai-sdk/file";
import { extractAudio } from "../pipeline/audio";
import { ResolvedElement } from "../resolved-element";
import type { AudioElementProps, SpeechProps, VargElement } from "../types";
import type { RenderContext } from "./context";
import { renderSpeech } from "./speech";
import { renderTalkingHead } from "./talking-head";
import { computeCacheKey } from "./utils";
import { renderVideo } from "./video";

/**
 * Render an audio element to a File inside the render pipeline.
 *
 * Mirrors `resolveAudioElement` (standalone path) but reuses the render
 * context: parent video/speech render through the normal renderers with
 * pendingFiles dedup, and audio extraction goes through ctx.backend.
 */
export async function renderAudio(
  element: VargElement<"audio">,
  ctx: RenderContext,
): Promise<File> {
  // Pre-resolved (awaited, or derived from a resolved speech parent)
  if (element instanceof ResolvedElement || element.meta?.file) {
    const file = (element.meta as { file: File }).file;
    ctx.generatedFiles.push(file);
    return file;
  }

  const props = element.props as AudioElementProps;

  if (props.src) {
    return props.src.startsWith("http")
      ? File.fromUrl(props.src)
      : File.fromPath(props.src);
  }

  const parent = props.parent as VargElement | undefined;
  if (!parent) {
    throw new Error("Audio element requires a 'parent' element or 'src'");
  }

  if (parent.type === "speech") {
    return renderSpeech(parent as VargElement<"speech">, ctx);
  }

  if (parent.type === "video" || parent.type === "talking-head") {
    // Dedup concurrent extractions of the same parent
    const cacheKeyStr = `audio-extract:${JSON.stringify(computeCacheKey(parent))}`;
    const pending = ctx.pendingFiles.get(cacheKeyStr);
    if (pending) return pending;

    const promise = (async () => {
      const videoFile =
        parent.meta?.file ??
        (parent.type === "video"
          ? await renderVideo(parent as VargElement<"video">, ctx)
          : await renderTalkingHead(
              parent as VargElement<"talking-head">,
              ctx,
            ));
      const audioFile = await extractAudio(videoFile, ctx.backend);
      ctx.generatedFiles.push(audioFile);
      return audioFile;
    })();

    ctx.pendingFiles.set(cacheKeyStr, promise);
    return promise;
  }

  throw new Error(
    `Audio element parent must be a video or speech element, got "${parent.type}"`,
  );
}

/** Volume for an audio element in the mix (inherits speech parent volume). */
export function audioMixVolume(element: VargElement<"audio">): number {
  const props = element.props as AudioElementProps;
  if (props.volume !== undefined) return props.volume;
  const parentProps = props.parent?.props as SpeechProps | undefined;
  return parentProps?.volume ?? 1;
}
