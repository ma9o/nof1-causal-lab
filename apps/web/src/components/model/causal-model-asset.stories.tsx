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
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { getEpisodeProgressQueryKey } from "@/lib/hooks/use-run-events";
import { CausalModelAssetView } from "./causal-model-asset";
import type { ActionTraceState } from "./conversation-pane";

const worker = setupWorker();

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
          useActionTrace={useStoryTrace}
          running={episode.data.running}
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
          "A harness-driven model viewer: exact API action names and short commit hashes above the graph, selected-action details below, and the selected action's log on the right. Failed actions keep their cross marker and attempt commit hash. The story walks from a question through structure, measurement and laws, a table import, a convergence warning, individual paired outcome paths with optional summaries, an edited-model simulation whose causal effect is unavailable, and a fit still running after it. Expand lineage to see both branches and inspect any action. All data is illustrative; statistics and differences are read from backend-shaped fixtures, and the viewer offers no write controls.",
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
      ["edit_model · c000002", "Model changes"],
      ["prepare_data · c000005", "Prepared data"],
      ["edit_model · c000006 · failed", "Edit model failed"],
      ["fit · c000008", "Parameter diagnostics"],
      ["simulate · c000009", "Simulation design"],
    ]) {
      await userEvent.click(canvas.getByRole("button", { name: label }));
      await expect(
        await canvas.findByRole("option", { name: label.replace(" · failed", ""), selected: true }),
      ).toBeInTheDocument();
      await expect(
        within(canvas.getByRole("complementary", { name: "Action log" })).getByText(
          label.replace(" · failed", ""),
        ),
      ).toBeVisible();
      await expect(await canvas.findByRole("region", { name: section })).toBeVisible();
      if (section === "Prepared data") {
        await expect(
          (await canvas.findAllByRole("img", { name: /prepared observations/ }))[0],
        ).toBeInTheDocument();
      }
      if (section === "Simulation design") {
        await expect(
          (await canvas.findAllByRole("img", { name: /recorded draws/ }))[0],
        ).toBeInTheDocument();
        const firstDraw = (await canvas.findAllByRole("spinbutton", { name: "First draw" }))[0];
        await userEvent.clear(firstDraw);
        await userEvent.type(firstDraw, "25");
        await userEvent.tab();
        await expect(await canvas.findByText(/25–48 of 100/)).toBeInTheDocument();
      }
    }
    await expect(canvas.queryByRole("log", { name: "fit · running" })).not.toBeInTheDocument();
    await userEvent.click(canvas.getByRole("button", { name: "simulate · c00000c · latest" }));
    await expect(
      await canvas.findByText(
        "This edited model has no committed production fit at this revision.",
      ),
    ).toBeVisible();
    await expect(canvas.getByRole("log", { name: "fit · running" })).toBeVisible();
    await userEvent.click(canvas.getByRole("button", { name: "Collapse lineage" }));
  },
};
