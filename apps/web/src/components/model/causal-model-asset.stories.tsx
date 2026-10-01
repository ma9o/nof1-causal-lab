import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { setupWorker } from "msw/browser";
import { http, HttpResponse } from "msw";
import { expect, userEvent, within } from "storybook/test";
import { useState, type ReactNode } from "react";
import {
  WORKBENCH_WORKSPACE,
  workbenchHandlers,
  workbenchQuestion,
  workbenchTraces,
} from "@/components/__fixtures__/workbench";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useModelSnapshot } from "@/lib/hooks/use-model-snapshot";
import { getStudyJournalQueryKey, readStudyJournal } from "@/lib/hooks/use-study-journal";
import { CausalModelAssetView } from "./causal-model-asset";
import type { ActionTraceState } from "./action-record";

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
  const journal = useQuery({
    queryKey: getStudyJournalQueryKey(WORKBENCH_WORKSPACE),
    queryFn: () => readStudyJournal(WORKBENCH_WORKSPACE),
  });
  if (journal.error) return <p role="alert">{journal.error.message}</p>;
  if (!journal.data) return <p role="status">Loading the workbench…</p>;
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
          attempts={journal.data.attempts}
          branches={journal.data.branches}
          dependencies={journal.data.dependencies}
          useActionTrace={useStoryTrace}
          running={journal.data.running}
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
          "A harness-driven model viewer: exact API action names and short commit hashes above the graph, model or selected-entity state below, and the selected action’s outcome and reasons on the right. Failed actions keep their cross marker and attempt commit hash. The story walks from a question through structure, measurement and laws, a table import, a convergence warning, individual paired outcome paths with optional summaries, an edited-model simulation whose causal effect is unavailable, and a fit still running after it. Links follow each action's arguments, including the alternative branch's fork; inspect any action. All data is illustrative; statistics and differences are read from backend-shaped fixtures, and the viewer offers no write controls.",
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
    // The data_diff links to the panel and the simulation it compared, not to the head.
    await canvas.findByRole("button", { name: "data_diff · c00000d" });
    await expect(canvasElement.querySelectorAll("path[data-argument]")).toHaveLength(14);
    for (const [label, section, state] of [
      ["edit_model · c000002", "Model changes", null],
      ["prepare_data · c000005", "Prepared data", "gad7 screening score"],
      ["edit_model · c000006 · failed", "Edit model failed", null],
      ["fit · c000008", "Parameter convergence", "Joint posterior"],
      ["simulate · c000009", null, "Simulation design"],
    ] as const) {
      await userEvent.click(canvas.getByRole("button", { name: label }));
      await expect(
        await within(canvas.getByRole("complementary", { name: "Action record" })).findByText(
          label.replace(" · failed", ""),
        ),
      ).toBeVisible();
      const record = within(canvas.getByRole("complementary", { name: "Action record" }));
      const details = within(canvas.getByRole("region", { name: "Model details" }));
      await expect(details.getByRole("heading", { name: "Model state" })).toBeVisible();
      if (section) await expect(await record.findByRole("region", { name: section })).toBeVisible();
      if (state) {
        await expect(await details.findByRole("region", { name: state })).toBeInTheDocument();
        await expect(record.queryByRole("region", { name: state })).not.toBeInTheDocument();
      }
      if (section === "Prepared data") {
        await expect(record.getByText(/Extraction incomplete/)).toBeVisible();
        await expect(canvas.queryByText("EXTRACTION_PARTIAL")).not.toBeInTheDocument();
        await expect(
          await details.findByRole("img", { name: /gad7_screening_score: prepared observations/ }),
        ).toBeInTheDocument();
        await expect(details.queryByText("Time coverage")).not.toBeInTheDocument();
      }
      if (section === "Parameter convergence") {
        await expect(
          record.getByText(/R-hat < 1.01 not met for 1 of \d+ parameters/),
        ).toBeVisible();
        await expect(
          details.queryByRole("region", { name: "Posterior predictive checks" }),
        ).not.toBeInTheDocument();
      }
      if (state === "Simulation design") {
        await expect(record.getByRole("log", { name: "Simulator log" })).toBeVisible();
        await expect(record.getByText("ACTION_COMPLETED")).toBeVisible();
        await expect(record.queryByRole("region")).not.toBeInTheDocument();
        const checks = canvas.getByRole("region", { name: "Simulation checks" });
        await expect(within(checks).getByText("internalizing symptom burden")).toBeInTheDocument();
        await expect(within(checks).getByText("gad7 screening score")).toBeInTheDocument();
        await expect(checks.scrollWidth).toBeLessThanOrEqual(checks.clientWidth);
        await expect(
          await details.findByRole("img", { name: /recorded paired effects/ }),
        ).toBeInTheDocument();
        await expect(
          await details.findByRole("img", { name: "internalizing_symptom_burden: recorded draws" }),
        ).toBeInTheDocument();
        await expect(
          await details.findByRole("img", { name: "phq9_screening_score: recorded draws" }),
        ).toBeInTheDocument();
        await userEvent.click(
          await within(canvas.getByRole("region", { name: "Causal graph" })).findByRole("button", {
            name: "internalizing symptom burden",
          }),
        );
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
    await expect(canvas.queryByRole("combobox", { name: "Details scope" })).not.toBeInTheDocument();
    await expect(canvas.queryByRole("log", { name: "fit · running" })).not.toBeInTheDocument();
    await userEvent.click(canvas.getByRole("button", { name: "simulate · c00000c · latest" }));
    await expect(await canvas.findByRole("log", { name: "fit · running" })).toBeVisible();
    await expect(
      within(canvas.getByRole("region", { name: "Model details" })).getByRole("heading", {
        name: "internalizing symptom burden",
      }),
    ).toBeVisible();
    await userEvent.click(canvas.getByRole("button", { name: "Show model state" }));
    await expect(
      await within(canvas.getByRole("region", { name: "Model details" })).findByText(
        "This edited model has no committed production fit at this revision.",
      ),
    ).toBeVisible();
    const record = within(canvas.getByRole("complementary", { name: "Action record" }));
    const details = within(canvas.getByRole("region", { name: "Model details" }));
    worker.use(
      http.get(
        `/api/studies/${WORKBENCH_WORKSPACE}/data-diff/:commit`,
        () => HttpResponse.json({ detail: "Comparison report unavailable" }, { status: 503 }),
        { once: true },
      ),
    );
    await userEvent.click(canvas.getByRole("button", { name: "data_diff · c00000d" }));
    await expect(await record.findByRole("alert")).toHaveTextContent(
      "Comparison report unavailable",
    );
    await expect(details.getByRole("alert")).toHaveTextContent("Comparison report unavailable");
    await expect(canvas.getByRole("region", { name: "Causal graph" })).toBeVisible();
    await userEvent.click(canvas.getByRole("button", { name: "simulate · c00000c · latest" }));
    await expect(await record.findByRole("log", { name: "fit · running" })).toBeVisible();
    await userEvent.click(canvas.getByRole("button", { name: "data_diff · c00000d" }));
    await expect(await record.findByRole("region", { name: "Data comparison" })).toBeVisible();
    await expect(record.getByText(/Panel from prepare_data · c000005/)).toBeVisible();
    await expect(record.getByText(/Observed values fall outside/)).toBeVisible();
    await expect(
      await details.findByRole("img", { name: "gad7_screening_score: data comparison" }),
    ).toBeInTheDocument();
    await userEvent.click(
      await within(canvas.getByRole("region", { name: "Causal graph" })).findByRole("button", {
        name: "gad7 screening score comparison",
      }),
    );
    await expect(details.getByRole("heading", { name: "gad7 screening score" })).toBeVisible();
    await expect(
      details.getByRole("table", { name: /statistic distributions/ }),
    ).toBeInTheDocument();
    await expect(canvas.queryByRole("log", { name: "fit · running" })).not.toBeInTheDocument();
    await userEvent.click(canvas.getByRole("button", { name: "simulate · c00000c · latest" }));
    await expect(await canvas.findByRole("log", { name: "fit · running" })).toBeVisible();
  },
};
