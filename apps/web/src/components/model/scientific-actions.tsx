"use client";

import {
  createModelClient,
  type ArtifactVersionInfo,
  type ModelSpec,
  type JsonObject,
  type ScientificActionId,
} from "@nof1-causal-lab/api-types";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { JsonViewer } from "@/components/ui/json-viewer";

const client = createModelClient();
const actions = [
  ["edit_model", "Edit model"],
  ["prepare_data", "Prepare data"],
  ["fit", "Fit"],
  ["simulate", "Simulate"],
] as const;
const fieldClass = "rounded border bg-background px-2 py-1 text-xs";

function RevisionSelect({
  label,
  versions,
  value,
  onChange,
}: {
  label: string;
  versions: ArtifactVersionInfo[];
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-xs">
      {label}
      <select
        aria-label={label}
        className={fieldClass}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        <option value={0}>Select revision</option>
        {versions.map((version) => (
          <option key={version.version} value={version.version}>
            v{version.version} ·{" "}
            {version.produced_by === "run:posterior" ? "fitted" : version.provenance}
          </option>
        ))}
      </select>
    </label>
  );
}

/** All statistical work and contract validation are owned by the Python actions. */
export function ScientificActions({
  workspaceId,
  currentVersion,
  panelVersion,
  rawVersion,
  busy,
  onRecipe,
}: {
  workspaceId: string;
  currentVersion: number;
  panelVersion: number;
  rawVersion: number;
  busy: boolean;
  onRecipe: (() => void) | null;
}) {
  const [action, setAction] = useState<ScientificActionId | null>(null);
  const [selectedModel, setModel] = useState<number | null>(null);
  const [selectedPanel, setPanel] = useState<number | null>(null);
  const [selectedRaw, setRaw] = useState<number | null>(null);
  const [compareWith, setCompareWith] = useState(0);
  const modelVersion = selectedModel ?? currentVersion;
  const dataVersion = selectedPanel ?? (action === "simulate" ? 0 : panelVersion);
  const sourceVersion = selectedRaw ?? rawVersion;
  const params = { path: { workspace_id: workspaceId } };
  const catalog = useQuery({
    queryKey: ["scientific-revisions", workspaceId, currentVersion, panelVersion, rawVersion],
    queryFn: async () => {
      const { data, error } = await client.GET("/api/episodes/{workspace_id}/revisions", {
        params,
      });
      if (error || !data) throw new Error(JSON.stringify(error));
      return data;
    },
    enabled: action !== null,
  });
  const definition = useQuery({
    queryKey: ["scientific-definition", workspaceId, modelVersion],
    queryFn: async () => {
      const { data, error } = await client.GET(
        "/api/episodes/{workspace_id}/revisions/model/{version}",
        { params: { path: { workspace_id: workspaceId, version: modelVersion } } },
      );
      if (error || !data) throw new Error(JSON.stringify(error));
      return data;
    },
    enabled: action !== null && modelVersion > 0,
  });
  const comparison = useQuery({
    queryKey: ["scientific-comparison", workspaceId, compareWith, modelVersion],
    queryFn: async () => {
      const { data, error } = await client.GET("/api/episodes/{workspace_id}/revisions/compare", {
        params: { ...params, query: { before: compareWith, after: modelVersion } },
      });
      if (error || !data) throw new Error(JSON.stringify(error));
      return data;
    },
    enabled: compareWith > 0 && modelVersion > 0 && action !== null,
  });
  const profile = useQuery({
    queryKey: ["data-profile", workspaceId, dataVersion],
    queryFn: async () => {
      const { data, error } = await client.GET(
        "/api/episodes/{workspace_id}/revisions/data-profile/{panel_version}",
        { params: { path: { workspace_id: workspaceId, panel_version: dataVersion } } },
      );
      if (error || !data) throw new Error(JSON.stringify(error));
      return data;
    },
    enabled: action === "prepare_data" && dataVersion > 0,
  });
  const error = catalog.error ?? definition.error ?? comparison.error;
  return (
    <div className="flex-none border-b px-6 py-2">
      <div className="flex items-center gap-2">
        {actions.map(([id, label]) => (
          <Button
            key={id}
            size="sm"
            variant={action === id ? "default" : "outline"}
            onClick={() => setAction(action === id ? null : id)}
            disabled={busy}
          >
            {label}
          </Button>
        ))}
        {onRecipe && (
          <details className="ml-auto text-xs">
            <summary className="cursor-pointer">Optional recipes</summary>
            <Button size="sm" variant="ghost" onClick={onRecipe} disabled={busy}>
              Build observational study
            </Button>
          </details>
        )}
      </div>
      {action && (
        <div className="mt-3 max-h-[55vh] space-y-3 overflow-auto">
          <div className="flex flex-wrap gap-4">
            <RevisionSelect
              label="Model"
              versions={catalog.data?.models ?? []}
              value={modelVersion}
              onChange={setModel}
            />
            <RevisionSelect
              label={action === "simulate" ? "Comparison observations (optional)" : "Observations"}
              versions={catalog.data?.panels ?? []}
              value={dataVersion}
              onChange={setPanel}
            />
            {action === "prepare_data" && (
              <RevisionSelect
                label="Source"
                versions={catalog.data?.raw_data ?? []}
                value={sourceVersion}
                onChange={setRaw}
              />
            )}
            <RevisionSelect
              label="Compare model with"
              versions={catalog.data?.models ?? []}
              value={compareWith}
              onChange={setCompareWith}
            />
          </div>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error.message}
            </p>
          )}
          {(modelVersion === 0 || definition.data) && (
            <ActionEditor
              key={`${action}:${modelVersion}:${dataVersion}:${sourceVersion}:${currentVersion}`}
              action={action}
              workspaceId={workspaceId}
              model={definition.data}
              modelVersion={modelVersion}
              currentVersion={currentVersion}
              panelVersion={dataVersion}
              rawVersion={sourceVersion}
            />
          )}
          {comparison.data && (
            <details open>
              <summary className="text-sm">
                Parameter decisions: v{compareWith} → v{modelVersion}
              </summary>
              <table className="w-full text-left text-xs">
                <thead>
                  <tr>
                    <th>Parameter</th>
                    <th>Change</th>
                    <th>Before</th>
                    <th>After</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.data.parameters.map((change) => (
                    <tr key={change.parameter_id}>
                      <td>{change.after?.name ?? change.before?.name}</td>
                      <td>{change.change}</td>
                      <td>
                        {change.before?.value ?? change.before?.distribution ?? "unspecified"}
                      </td>
                      <td>{change.after?.value ?? change.after?.distribution ?? "unspecified"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <JsonViewer data={comparison.data as unknown as JsonObject} />
            </details>
          )}
          {action === "prepare_data" && (
            <details>
              <summary className="text-sm">Data profile</summary>
              {profile.error ? (
                <p>{profile.error.message}</p>
              ) : (
                <JsonViewer data={(profile.data ?? {}) as unknown as JsonObject} />
              )}
            </details>
          )}
        </div>
      )}
    </div>
  );
}

function ActionEditor({
  action,
  workspaceId,
  model,
  modelVersion,
  currentVersion,
  panelVersion,
  rawVersion,
}: {
  action: ScientificActionId;
  workspaceId: string;
  model: ModelSpec | undefined;
  modelVersion: number;
  currentVersion: number;
  panelVersion: number;
  rawVersion: number;
}) {
  const initial =
    action === "edit_model"
      ? { action, expected_version: currentVersion, model: model ?? {} }
      : action === "prepare_data"
        ? { action, source: "raw_data", model_version: modelVersion, raw_data_version: rawVersion }
        : action === "fit"
          ? { action, model_version: modelVersion, panel_version: panelVersion, settings: {} }
          : {
              action,
              model_version: modelVersion,
              ...(panelVersion > 0 ? { comparison_panel_version: panelVersion } : {}),
              design: {
                kind: "trajectory",
                times: [0, 1, 2, 3, 4, 5, 6, 7],
                draws: 100,
                seed: 0,
                initial_state: "new_study",
                process_noise: true,
                observation_noise: true,
                context: "exploration",
              },
            };
  const [text, setText] = useState(JSON.stringify(initial, null, 2));
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<unknown>(null);
  const [error, setError] = useState<string | null>(null);
  const cache = useQueryClient();
  async function submit() {
    setRunning(true);
    setError(null);
    try {
      const body = JSON.parse(text);
      if (body.action !== action) throw new Error("Use the selected action's contract");
      const { data, error } = await client.POST("/api/episodes/{workspace_id}/actions", {
        params: { path: { workspace_id: workspaceId } },
        body,
      });
      if (error) throw new Error(JSON.stringify(error));
      setResult(data);
      if (data?.status !== "applied")
        throw new Error(data?.reason ?? data?.error_message ?? "Action did not apply");
      await cache.invalidateQueries({
        predicate: (query) =>
          [
            "model-snapshot",
            "scientific-revisions",
            "scientific-comparison",
            "data-profile",
            "analysis",
          ].includes(String(query.queryKey[0])),
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setRunning(false);
    }
  }
  return (
    <div className="space-y-2">
      <p className="text-xs text-muted-foreground">
        {action === "edit_model"
          ? "Edit constructs, measurements, mechanisms, fixed values and laws together. Submission saves a revision and refreshes applicable checks."
          : action === "prepare_data"
            ? "Import uploaded files or prepare observations using the selected model's measurement definitions."
            : action === "fit"
              ? "Condition the selected definition on the selected observations. To refit an earlier definition, select its revision above."
              : "Set the schedule, initial state, noise, interventions and comparisons for this simulation. A retained start requires state_time matching the first design time. Comparison observations require their prepared time grid; use comparison_time_offset to align a different origin."}
      </p>
      {action === "prepare_data" && (
        <Button
          variant="outline"
          size="sm"
          onClick={() => setText(JSON.stringify({ action, source: "files" }, null, 2))}
        >
          Use uploaded files
        </Button>
      )}
      <label className="block text-xs">
        Action inputs
        <textarea
          aria-label="Action inputs"
          className="mt-1 h-48 w-full rounded border bg-background p-2 font-mono text-xs"
          value={text}
          onChange={(event) => setText(event.target.value)}
          spellCheck={false}
        />
      </label>
      <Button size="sm" onClick={submit} disabled={running}>
        {running ? "Running…" : "Submit"}
      </Button>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {result != null && (
        <details open>
          <summary className="text-sm">Result and checks</summary>
          <JsonViewer data={result as JsonObject} />
        </details>
      )}
    </div>
  );
}
