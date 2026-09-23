"use client";

import { cn } from "@/lib/utils";
import { StatTooltip } from "./stat-tooltip";
import { useDataTable, COL_MIN_WIDTH } from "@/lib/tables/use-data-table";

interface DataTableProps<T extends object> {
  rows: T[];
  maxHeight?: string;
  columnTooltips?: Record<string, string>;
}

export function DataTable<T extends object>({
  rows,
  maxHeight = "max-h-64",
  columnTooltips,
}: DataTableProps<T>) {
  "use no memo"; // TODO: remove when TanStack Virtual supports React Compiler
  const { columns, parentRef, virtualizer, focusedRowIndex, containerProps, tableWidth } =
    useDataTable(rows);
  if (rows.length === 0) return null;

  return (
    <div
      ref={parentRef}
      className={cn(maxHeight, "overflow-auto rounded-md border")}
      {...containerProps}
    >
      <div style={{ minWidth: tableWidth }}>
        {/* Sticky header — divs with ARIA roles needed for virtualized layout */}
        {/* biome-ignore lint/a11y/useFocusableInteractive: virtualized table uses divs with ARIA roles */}
        {/* biome-ignore lint/a11y/useSemanticElements: virtualized table uses divs with ARIA roles */}
        <div className="sticky top-0 z-10 flex border-b bg-background" role="row">
          {columns.map((col) => (
            <div
              key={col}
              className="py-1 px-3 text-xs font-medium text-muted-foreground capitalize truncate"
              style={{ minWidth: COL_MIN_WIDTH, flex: 1 }}
              // biome-ignore lint/a11y/useSemanticElements: virtualized table requires div-based layout
              role="columnheader"
            >
              <span className="inline-flex items-center gap-1">
                {col.replace(/_/g, " ")}
                {columnTooltips?.[col] && <StatTooltip explanation={columnTooltips[col]} />}
              </span>
            </div>
          ))}
        </div>

        {/* Virtualized body */}
        {/* biome-ignore lint/a11y/useSemanticElements: virtualized table uses divs with ARIA roles */}
        <div style={{ height: virtualizer.getTotalSize(), position: "relative" }} role="rowgroup">
          {virtualizer.getVirtualItems().map((vi) => {
            const row = rows[vi.index] as Record<string, unknown>;
            return (
              // biome-ignore lint/a11y/useFocusableInteractive: virtualized table uses divs with ARIA roles
              <div
                key={vi.index}
                className={cn(
                  "absolute left-0 right-0 flex border-b border-border/40 hover:bg-muted/50",
                  focusedRowIndex === vi.index && "ring-2 ring-ring ring-inset",
                )}
                style={{
                  height: vi.size,
                  transform: `translateY(${vi.start}px)`,
                }}
                // biome-ignore lint/a11y/useSemanticElements: virtualized table uses divs with ARIA roles
                role="row"
              >
                {columns.map((col) => (
                  // biome-ignore lint/a11y/useFocusableInteractive: virtualized table uses divs with ARIA roles
                  <div
                    key={col}
                    className="py-1 px-3 text-xs text-muted-foreground truncate leading-5"
                    style={{ minWidth: COL_MIN_WIDTH, flex: 1 }}
                    // biome-ignore lint/a11y/useSemanticElements: virtualized table requires div-based layout
                    role="gridcell"
                  >
                    {row[col] == null ? (
                      <span className="text-muted-foreground/40 italic">N/A</span>
                    ) : typeof row[col] === "boolean" ? (
                      row[col] ? (
                        "true"
                      ) : (
                        "false"
                      )
                    ) : (
                      String(row[col])
                    )}
                  </div>
                ))}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
