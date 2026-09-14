/** Generate model read operations from FastAPI, reusing the canonical domain declarations. */
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import openapiTS, { astToString } from "openapi-typescript";
import ts from "typescript";

const root = resolve(import.meta.dirname, "..");
const schema = JSON.parse(readFileSync(resolve(root, "schemas/openapi.json"), "utf8"));
const models = readFileSync(resolve(root, "src/generated/models.ts"), "utf8");
const names = new Set([...models.matchAll(/export (?:interface|type) (\w+)/g)].map((m) => m[1]));
// The client covers the model read API. Write inputs keep their own OpenAPI validation semantics.
schema.paths = Object.fromEntries(
  Object.entries(schema.paths).filter(([path]) =>
    /^\/api\/episodes\/\{workspace_id\}\/model(?:\/|$)/.test(path),
  ),
);

// Keep only definitions reachable from these operations.
const referenced = new Set<string>();
function visit(value: unknown): void {
  if (!value || typeof value !== "object") return;
  if ("$ref" in value && typeof value.$ref === "string") {
    const name = value.$ref.replace("#/components/schemas/", "");
    if (!referenced.has(name)) {
      referenced.add(name);
      visit(schema.components.schemas[name]);
    }
  }
  for (const child of Object.values(value)) visit(child);
}
visit(schema.paths);
schema.components.schemas = Object.fromEntries(
  Object.entries(schema.components.schemas).filter(([name]) => referenced.has(name)),
);

const ast = await openapiTS(schema, {
  inject: 'import type * as Domain from "./models";',
  transform(_value, metadata) {
    const name = metadata.path?.match(/^#\/components\/schemas\/([^/]+)$/)?.[1];
    const canonical = name?.replace(/-(?:Input|Output)$/, "");
    if (canonical && names.has(canonical)) {
      return ts.factory.createTypeReferenceNode(`Domain.${canonical}`);
    }
  },
});
const content = `/** AUTO-GENERATED from FastAPI OpenAPI. Run bun run codegen. */\n${astToString(ast)}`;
const output = resolve(root, "src/generated/model-api.ts");
if (process.argv.includes("--check")) {
  if (readFileSync(output, "utf8") !== content) {
    throw new Error("Model API client types are stale. Run bun run codegen.");
  }
  console.log("Model API client types checked.");
} else {
  writeFileSync(output, content);
  console.log("Generated model API client types.");
}
