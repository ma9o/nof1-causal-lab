import Link from "next/link";
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** Temporary navigation between two presentations of the same study. */
export function WorkspaceVersionFrame({
  workspaceId,
  version,
  children,
}: {
  workspaceId: string;
  version: "v1" | "v2";
  children: ReactNode;
}) {
  return (
    <div className={cn(version === "v2" && "md:flex md:h-dvh md:flex-col md:overflow-hidden")}>
      <nav
        aria-label="Interface version"
        className="flex h-10 flex-none items-center gap-1 border-b bg-card px-4"
      >
        {(["v1", "v2"] as const).map((target) => (
          <Link
            key={target}
            href={`/${target}/${encodeURIComponent(workspaceId)}`}
            aria-current={target === version ? "page" : undefined}
            className={cn(
              "rounded-md px-2 py-1 text-xs focus-visible:outline-2 focus-visible:outline-ring",
              target === version
                ? "bg-muted font-medium text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {target === "v1" ? "v1 · Pipeline" : "v2 · Model"}
          </Link>
        ))}
      </nav>
      <div
        className={cn(
          version === "v2" && "md:min-h-0 md:flex-1 [&>div]:md:h-full [&>div]:md:min-h-0",
        )}
      >
        {children}
      </div>
    </div>
  );
}
