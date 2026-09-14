import { getToolServerUrl } from "@/lib/runtime-urls";
import type { ArtifactEnvelope, ArtifactId } from "@nof1-causal-lab/api-types";
import { ARTIFACT_FILE_SPECS } from "@nof1-causal-lab/api-types";

export type { ArtifactId } from "@nof1-causal-lab/api-types";

type FileKind = "json" | "parquet";

export class ArtifactNotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ArtifactNotFoundError";
  }
}

const TOOL_SERVER = getToolServerUrl();

function artifactFileName(artifactId: ArtifactId, kind: FileKind, key: string): string {
  const filename = ARTIFACT_FILE_SPECS[artifactId][kind]?.[key];
  if (!filename) {
    throw new ArtifactNotFoundError(`${artifactId} has no declared ${kind} file '${key}'`);
  }
  return filename;
}

async function fetchArtifact(
  workspaceId: string,
  artifactId: ArtifactId,
  version?: number,
): Promise<ArtifactEnvelope> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${encodeURIComponent(workspaceId)}/artifacts/${encodeURIComponent(
      artifactId,
    )}${version == null ? "" : `?version=${version}`}`,
    { cache: "no-store" },
  );
  if (response.status === 404) {
    throw new ArtifactNotFoundError(await response.text());
  }
  if (!response.ok) {
    throw new Error(`Artifact facade error ${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<ArtifactEnvelope>;
}

async function fetchArtifactFile(
  workspaceId: string,
  artifactId: ArtifactId,
  filename: string,
): Promise<Uint8Array> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${encodeURIComponent(workspaceId)}/artifacts/${encodeURIComponent(
      artifactId,
    )}/files/${encodeURIComponent(filename)}`,
    { cache: "no-store" },
  );
  if (response.status === 404) {
    throw new ArtifactNotFoundError(await response.text());
  }
  if (!response.ok) {
    throw new Error(`Artifact facade file error ${response.status}: ${await response.text()}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

export async function readArtifactJson<T>(
  workspaceId: string,
  artifactId: ArtifactId,
  key: string,
  version?: number,
): Promise<T> {
  const filename = artifactFileName(artifactId, "json", key);
  const artifact = await fetchArtifact(workspaceId, artifactId, version);
  if (!(filename in artifact.payload)) {
    throw new ArtifactNotFoundError(`${artifactId} has no payload file '${filename}'`);
  }
  return artifact.payload[filename] as T;
}

export async function readArtifactBinary(
  workspaceId: string,
  artifactId: ArtifactId,
  kind: Exclude<FileKind, "json">,
  key: string,
): Promise<{ data: Uint8Array; filename: string }> {
  const filename = artifactFileName(artifactId, kind, key);
  return {
    data: await fetchArtifactFile(workspaceId, artifactId, filename),
    filename,
  };
}
