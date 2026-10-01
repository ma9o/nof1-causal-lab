import type { ReactNode } from "react";

/** On desktop the workbench fills the viewport; its panes scroll independently. */
export default function ModelLayout({ children }: { children: ReactNode }) {
  return (
    <div className="md:flex md:h-dvh md:flex-col md:overflow-hidden">
      <div className="md:min-h-0 md:flex-1 [&>div]:md:h-full [&>div]:md:min-h-0">{children}</div>
    </div>
  );
}
