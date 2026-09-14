import type { PipelineSectionId } from "@nof1-causal-lab/api-types";

export function isMockMode(): boolean {
  const v = process.env.NEXT_PUBLIC_MOCK_DATA;
  return !!v && v !== "false";
}

/** Returns the fixture user ID, e.g. "DEFAULT". */
export function getMockFixture(): string {
  const v = process.env.NEXT_PUBLIC_MOCK_DATA;
  if (!v || v === "true") return "DEFAULT";
  const fixture = v.toUpperCase();
  return fixture === "DEMO_HEALTH" ? "DEMO" : fixture;
}

export interface MockEventHandler {
  onTransitionStart: (artifactId: PipelineSectionId) => void;
  onTransitionComplete: (artifactId: PipelineSectionId) => void;
}

export function simulatePipelineEvents(
  handlers: MockEventHandler,
  transitionOrder: readonly PipelineSectionId[],
): () => void {
  for (const artifactId of transitionOrder) {
    handlers.onTransitionStart(artifactId);
    handlers.onTransitionComplete(artifactId);
  }

  return () => {};
}
