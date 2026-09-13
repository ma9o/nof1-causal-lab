"use client";

import type { LatentStructureArtifact } from "@nof1-causal-lab/api-types";
import { useState } from "react";
import { ConstructDetailPanel } from "@/components/analysis-widgets/latent-structure/construct-detail-panel";
import { EdgeList } from "@/components/analysis-widgets/latent-structure/edge-list";
import { StructureDag } from "@/components/dag/structure-dag";

export default function LatentStructureView({ data }: { data: LatentStructureArtifact }) {
  const [selectedConstruct, setSelectedConstruct] = useState<string | null>(null);
  const selected = data.latent_structure.constructs.find((c) => c.name === selectedConstruct);

  return (
    <div className="space-y-4">
      <StructureDag
        constructs={data.latent_structure.constructs}
        outcomeId={data.latent_structure.default_outcome?.id}
        edges={data.latent_structure.edges}
        onNodeClick={setSelectedConstruct}
      />
      {selected && (
        <ConstructDetailPanel
          construct={selected}
          isOutcome={selected.id === data.latent_structure.default_outcome?.id}
        />
      )}
      <EdgeList constructs={data.latent_structure.constructs} edges={data.latent_structure.edges} />
    </div>
  );
}
