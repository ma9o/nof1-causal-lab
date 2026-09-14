import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ConstructSpec } from "@nof1-causal-lab/api-types";
import { Star } from "lucide-react";

interface ConstructDetailPanelProps {
  construct: ConstructSpec;
  isOutcome?: boolean;
}

export function ConstructDetailPanel({ construct, isOutcome = false }: ConstructDetailPanelProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <CardTitle className="text-base">{construct.name}</CardTitle>
          {isOutcome && <Star className="h-4 w-4 fill-foreground/75 text-foreground/75" />}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">{construct.description}</p>

        <div className="flex flex-wrap gap-2">
          <Badge variant={construct.role === "endogenous" ? "default" : "secondary"}>
            {construct.role}
          </Badge>
          <Badge variant="outline">{construct.temporal_status.replace("_", " ")}</Badge>
          {isOutcome && <Badge variant="warning">outcome</Badge>}
        </div>
      </CardContent>
    </Card>
  );
}
