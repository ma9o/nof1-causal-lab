"use client";

import type { JsonObject, JsonValue } from "@nof1-causal-lab/api-types";
import { useState } from "react";

function JsonEntry({ name, value }: { name: string; value: JsonValue }) {
  const [expanded, setExpanded] = useState(false);
  if (value === null || typeof value !== "object") {
    return (
      <div className="break-words py-0.5">
        <span className="text-muted-foreground">{name}: </span>
        {JSON.stringify(value)}
      </div>
    );
  }

  const size = Object.keys(value).length;
  return (
    <details open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
      <summary className="cursor-pointer rounded py-1 hover:bg-muted">
        {name}{" "}
        <span className="text-muted-foreground">
          ({size} {Array.isArray(value) ? "items" : "fields"})
        </span>
      </summary>
      {expanded && (
        <div className="ml-2 border-l pl-3">
          {Object.entries(value).map(([key, child]) => (
            <JsonEntry key={key} name={key} value={child} />
          ))}
        </div>
      )}
    </details>
  );
}

export function JsonViewer({ data }: { data: JsonObject }) {
  return (
    <div className="max-h-96 overflow-auto rounded-md border bg-muted/20 p-3 font-mono text-xs">
      {Object.entries(data).map(([key, value]) => (
        <JsonEntry key={key} name={key} value={value} />
      ))}
    </div>
  );
}
