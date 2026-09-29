import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { setupWorker } from "msw/browser";
import { expect, userEvent, within } from "storybook/test";
import { useState, type ReactNode } from "react";
import {
  WORKBENCH_WORKSPACE,
  workbenchHandlers,
  workbenchQuestion,
  workbenchTraces,
} from "@/components/__fixtures__/workbench";
import { TooltipProvider } from "@/components/ui/tooltip";
import { getEpisodeProgress } from "@/lib/api/analysis";
import type { PipelineProgress } from "@/lib/hooks/pipeline-progress";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { getEpisodeProgressQueryKey } from "@/lib/hooks/use-run-events";
import { CausalModelAssetView } from "./causal-model-asset";
import type { ActionTraceState } from "./conversation-pane";

const worker = setupWorker();
const progress: PipelineProgress = {
  artifacts: Object.fromEntries(
    TRANSITIONS.map((section) => [section.id, "completed"]),
  ) as PipelineProgress["artifacts"],
  timings: {},
  transitionErrors: {},
  staleArtifactsByProducer: {},

  transitionOrder: TRANSITIONS.map((section) => section.id),
  runningTransitions: [],
  isComplete: true,
  isFailed: false,
};

function useStorySnapshot(commitId: string) {
  return useModelSnapshot(WORKBENCH_WORKSPACE, commitId);
}
function useStoryTrace(seq: number, enabled: boolean): ActionTraceState {
  const trace = workbenchTraces.get(seq);
  return enabled && trace ? { status: "ready", trace } : { status: "absent" };
}
function StoryProviders({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <TooltipProvider>{children}</TooltipProvider>
    </QueryClientProvider>
  );
}
function WorkbenchStory() {
  const episode = useQuery({
    queryKey: getEpisodeProgressQueryKey(WORKBENCH_WORKSPACE),
    queryFn: () => getEpisodeProgress(WORKBENCH_WORKSPACE),
  });
  if (episode.error) return <p role="alert">{episode.error.message}</p>;
  if (!episode.data) return <p role="status">Loading the workbench…</p>;
  return (
    <>
      <div
        role="note"
        className="flex h-6 items-center justify-end border-b px-5 text-[10px] text-muted-foreground"
      >
        Illustrative data
      </div>
      <div className="md:h-[calc(100dvh-24px)] [&>div]:md:h-full [&>div]:md:min-h-0">
        <CausalModelAssetView
          workspaceId={WORKBENCH_WORKSPACE}
          question={workbenchQuestion}
          useSnapshot={useStorySnapshot}
          transitions={episode.data.transitions}
          branches={episode.data.branches}
          progress={progress}
          useActionTrace={useStoryTrace}
        />
      </div>
    </>
  );
}

const meta = {
  title: "V2/Model/Workbench",
  component: WorkbenchStory,
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "A harness-driven model viewer: meaningful action summaries and compact failures above the graph, selected-version details below, and a persistent action log on the right. The story walks from a question through structure, measurement and laws, a table import, a convergence warning, paired outcome trajectories with 95% bands, and an edited-model simulation whose causal effect is unavailable. Expand lineage to see both branches and inspect any version or failed attempt. All data is illustrative; statistics and differences are read from backend-shaped fixtures, and the viewer offers no write controls.",
      },
    },
  },
  decorators: [
    (Story) => (
      <StoryProviders>
        <Story />
      </StoryProviders>
    ),
  ],
  beforeEach: async () => {
    worker.resetHandlers(...workbenchHandlers());
    await worker.start({
      quiet: true,
      onUnhandledRequest: (request, print) => {
        if (new URL(request.url).pathname.startsWith("/api/")) print.error();
      },
    });
    return () => worker.stop();
  },
} satisfies Meta<typeof WorkbenchStory>;
export default meta;
type Story = StoryObj<typeof meta>;

// Keep ONE comprehensive workbench story. Add new features and interactions to
// this scenario instead of creating separate feature, timeline, or state stories.
export const Complete: Story = {
  play: async ({ canvasElement }) => {
    const canvas = within(canvasElement);
    await userEvent.click(await canvas.findByRole("button", { name: "Expand lineage" }));
    for (const [label, section] of [
      ["Edit model · version 2", "Model changes"],
      ["Prepare data · version 5", "Prepared data"],
      ["Edit model · failed attempt 6", "Edit model failed"],
      ["Fit · version 8", "Parameter diagnostics"],
      ["Simulate · version 9", "Simulation design"],
    ]) {
      await userEvent.click(canvas.getByRole("button", { name: label }));
      await expect(await canvas.findByRole("region", { name: section })).toBeVisible();
    }
    await userEvent.click(canvas.getByRole("button", { name: "Simulate · version 12 · latest" }));
    await expect(
      await canvas.findByText(
        "This edited model has no committed production fit at this revision.",
      ),
    ).toBeVisible();
    await userEvent.click(canvas.getByRole("button", { name: "Collapse lineage" }));
  },
};
