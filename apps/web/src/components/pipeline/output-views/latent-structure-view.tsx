"use client";

import { modelConstructs } from "@/lib/model-accessors";
import type { ModelSpec } from "@nof1-causal-lab/api-types";
import { useState } from "react";
import { ConstructDetailPanel } from "@/components/analysis-widgets/latent-structure/construct-detail-panel";
import { EdgeList } from "@/components/analysis-widgets/latent-structure/edge-list";
import { StructureDag } from "@/components/dag/structure-dag";

export default function LatentStructureView({ data }: { data: ModelSpec }) {
  const [selectedConstruct, setSelectedConstruct] = useState<string | null>(null);
  const selected = modelConstructs(data).find((c) => c.name === selectedConstruct);

  return (
    <div className="space-y-4">
      <StructureDag
        constructs={modelConstructs(data)}
        outcomeId={data.default_outcome?.id}
        edges={data.edges}
        onNodeClick={setSelectedConstruct}
      />
      {selected && (
        <ConstructDetailPanel
          construct={selected}
          isOutcome={selected.id === data.default_outcome?.id}
        />
      )}
      <EdgeList constructs={modelConstructs(data)} edges={data.edges} />
    </div>
  );
}
