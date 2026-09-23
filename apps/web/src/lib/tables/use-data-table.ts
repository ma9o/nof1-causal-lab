"use client";

import { useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useTableKeyboardNav } from "./use-table-keyboard-nav";

const ROW_HEIGHT = 28;
export const COL_MIN_WIDTH = 120;

export function useDataTable<T extends object>(rows: T[]) {
  "use no memo";
  const columns = useMemo(() => {
    if (rows.length === 0) return [];
    const firstRow = rows[0] as Record<string, unknown>;
    const allKeys = Object.keys(firstRow);
    return allKeys.filter((key) =>
      rows.some((row) => (row as Record<string, unknown>)[key] != null),
    );
  }, [rows]);

  const parentRef = useRef<HTMLDivElement>(null);

  // eslint-disable-next-line react-hooks/incompatible-library -- This hook and its renderers opt out with "use no memo".
  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
  });

  const { focusedRowIndex, containerProps } = useTableKeyboardNav(rows.length);

  const tableWidth = Math.max(columns.length * COL_MIN_WIDTH, 0);

  return { columns, parentRef, virtualizer, focusedRowIndex, containerProps, tableWidth };
}
