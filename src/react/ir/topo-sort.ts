import type { Step } from "./types";

/**
 * Sort steps topologically: dependencies before dependents.
 * Stable for independent steps (preserves discovery order).
 *
 * @throws Error on circular dependencies.
 */
export function topoSort(steps: Step[]): Step[] {
  const result: Step[] = [];
  const visited = new Set<Step>();
  const visiting = new Set<Step>();

  function visit(step: Step, chain: string[]) {
    if (visited.has(step)) return;
    if (visiting.has(step)) {
      throw new Error(
        `Circular dependency detected: ${[...chain, step.id].join(" -> ")}`,
      );
    }
    visiting.add(step);
    for (const dep of step.dependsOn) {
      visit(dep, [...chain, step.id]);
    }
    visiting.delete(step);
    visited.add(step);
    result.push(step);
  }

  for (const step of steps) {
    visit(step, []);
  }

  return result;
}
