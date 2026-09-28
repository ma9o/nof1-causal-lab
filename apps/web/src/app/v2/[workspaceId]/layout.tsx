import type { ReactNode } from "react";
import { WorkspaceVersionFrame } from "@/components/workspace-version-frame";

export default async function ModelLayout({
  params,
  children,
}: {
  params: Promise<{ workspaceId: string }>;
  children: ReactNode;
}) {
  const { workspaceId } = await params;
  return (
    <WorkspaceVersionFrame workspaceId={workspaceId} version="v2">
      {children}
    </WorkspaceVersionFrame>
  );
}
