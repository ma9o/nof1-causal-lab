/**
 * Generate TypeScript types from JSON Schema exported by Python.
 *
 * Usage:
 *   cd packages/api-types
 *   bun run scripts/generate.ts
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, relative, resolve } from "node:path";
import { compile } from "json-schema-to-typescript";
import ts from "typescript";

// biome-ignore lint/suspicious/noExplicitAny: JSON Schema nodes are inherently untyped
type JsonSchema = any;

const ROOT = dirname(dirname(resolve(import.meta.filename)));
const SCHEMA_PATH = resolve(ROOT, "schemas", "contracts.json");
const METADATA_PATH = resolve(ROOT, "schemas", "metadata.json");
const OUTPUT_PATH = resolve(ROOT, "src", "generated", "models.ts");
const METADATA_OUTPUT_PATH = resolve(ROOT, "src", "generated", "metadata.ts");
const checkOnly = process.argv.includes("--check");
const changedPaths: string[] = [];

function readExisting(path: string): string | null {
  try {
    return readFileSync(path, "utf-8");
  } catch (error) {
    if (error && typeof error === "object" && "code" in error && error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

function writeOrCheck(outputPath: string, content: string): void {
  if (checkOnly) {
    if (readExisting(outputPath) !== content) {
      changedPaths.push(relative(ROOT, outputPath));
    }
    return;
  }

  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, content);
}

/**
 * Reduce a schema node to just its `$ref` if it has one.
 *
 * Pydantic emits `{"$ref": "#/$defs/Foo", "description": "..."}` for
 * fields with doc-strings. The sibling `description` (or `title`, `default`,
 * etc.) next to `$ref` causes json-schema-to-typescript to treat it as a
 * distinct anonymous type — generating duplicates like `LatentStructure1`.
 * Per JSON Schema 2020-12, `$ref` siblings are valid but the TS codegen
 * library doesn't handle them well, so we strip them.
 */
function collapseRefs(schema: JsonSchema): JsonSchema {
  if (typeof schema !== "object" || schema === null) return schema;
  if (Array.isArray(schema)) return schema.map(collapseRefs);

  // If this object has a $ref, keep only the $ref
  if ("$ref" in schema) {
    return { $ref: schema.$ref };
  }

  const result: JsonSchema = {};
  for (const [key, value] of Object.entries(schema)) {
    result[key] = collapseRefs(value);
  }
  return result;
}

/** json-schema-to-typescript ignores sibling properties beside oneOf/anyOf.
 * Express their JSON Schema conjunction as an explicit TypeScript intersection.
 */
function intersectUnionProperties(schema: JsonSchema): JsonSchema {
  if (typeof schema !== "object" || schema === null) return schema;
  if (Array.isArray(schema)) return schema.map(intersectUnionProperties);

  const result = Object.fromEntries(
    Object.entries(schema).map(([key, value]) => [key, intersectUnionProperties(value)]),
  );
  const keyword = "oneOf" in result ? "oneOf" : "anyOf";
  if (!("properties" in result) || !(keyword in result)) return result;

  const { properties, required, additionalProperties, oneOf, anyOf, allOf = [], ...rest } = result;
  return {
    ...rest,
    allOf: [
      ...allOf,
      { type: "object", properties, required, additionalProperties },
      ...(oneOf ? [{ oneOf }] : []),
      ...(anyOf ? [{ anyOf }] : []),
    ],
  };
}

/**
 * Strip field-level "title" from JSON Schema properties.
 *
 * Pydantic adds "title": "Field Name" to every field, which causes
 * json-schema-to-typescript to generate named type aliases for each
 * field (e.g., `type RHat = number | number[]`). This makes the
 * generated types hard to use with generic TS libraries like tanstack-table.
 *
 * We keep titles on top-level $defs (the actual model names) but strip
 * them from individual properties.
 */
function stripFieldTitles(schema: JsonSchema, isTopLevel = true): JsonSchema {
  if (typeof schema !== "object" || schema === null) return schema;

  if (Array.isArray(schema)) {
    return schema.map((item) => stripFieldTitles(item, false));
  }

  const result: JsonSchema = {};
  for (const [key, value] of Object.entries(schema)) {
    if (key === "properties" && typeof value === "object" && value !== null) {
      // Strip titles from property definitions
      const cleanProps: JsonSchema = {};
      for (const [propName, propSchema] of Object.entries(value as Record<string, JsonSchema>)) {
        const cleaned = { ...propSchema };
        delete cleaned.title;
        cleanProps[propName] = stripFieldTitles(cleaned, false);
      }
      result[key] = cleanProps;
    } else if (key === "$defs" && typeof value === "object" && value !== null) {
      // Keep titles on $defs (model-level names) but recurse into their contents
      const cleanDefs: JsonSchema = {};
      for (const [defName, defSchema] of Object.entries(value as Record<string, JsonSchema>)) {
        cleanDefs[defName] = stripFieldTitles(defSchema, true);
      }
      result[key] = cleanDefs;
    } else if (key === "items") {
      // Recurse into array items but strip their title
      const cleaned = typeof value === "object" ? { ...value } : value;
      if (typeof cleaned === "object" && cleaned !== null && !isTopLevel) {
        delete cleaned.title;
      }
      result[key] = stripFieldTitles(cleaned, false);
    } else if (key === "anyOf" || key === "oneOf") {
      // Strip titles from union members
      result[key] = (value as JsonSchema[]).map((item: JsonSchema) => {
        const cleaned = typeof item === "object" ? { ...item } : item;
        if (typeof cleaned === "object" && cleaned !== null) {
          delete cleaned.title;
        }
        return stripFieldTitles(cleaned, false);
      });
    } else {
      result[key] = value;
    }
  }

  return result;
}

function generateMetadata(): void {
  const metadata = JSON.parse(readFileSync(METADATA_PATH, "utf-8"));
  const byDist = metadata.observationHyperparametersByDistribution;
  const lines: string[] = [
    "/* eslint-disable */",
    "/**",
    " * AUTO-GENERATED — DO NOT EDIT",
    " *",
    " * Generated from Python distribution catalog via:",
    " *   cd apps/data-pipeline && uv run python -m scripts.codegen.export_api",
    " *   cd packages/api-types && bun run scripts/generate.ts",
    " *",
    " * Source of truth: apps/data-pipeline/src/nof1_causal_lab/distributions.py",
    " */",
    "",
    'import type { ArtifactId } from "./models";',
    `export const ARTIFACT_IDS = ${JSON.stringify(metadata.artifactIds)} as const satisfies readonly ArtifactId[];`,
    "",
    `const _OBS_HYPERS_BY_DIST = ${JSON.stringify(byDist, null, 2)} as const;`,
    "",
    "export type ObservationHyperparameter =",
    "  typeof _OBS_HYPERS_BY_DIST[keyof typeof _OBS_HYPERS_BY_DIST][number];",
    "",
    "export const OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION: Partial<",
    "  Record<string, readonly ObservationHyperparameter[]>",
    "> = _OBS_HYPERS_BY_DIST;",
    "",
  ];
  writeOrCheck(METADATA_OUTPUT_PATH, lines.join("\n"));
  if (!checkOnly) {
    console.log(`Generated metadata → ${METADATA_OUTPUT_PATH}`);
  }
}

/** Preserve the Python value interface in generated outputs, including map keys. */
function readonlyOutputs(source: string, schema: JsonSchema): string {
  const file = ts.createSourceFile("models.ts", source, ts.ScriptTarget.Latest, true);
  const transformed = ts.transform(file, [
    (context) => {
      const visit: ts.Visitor = (node) => {
        const child = ts.visitEachChild(node, visit, context);
        if (ts.isArrayTypeNode(child) || ts.isTupleTypeNode(child)) {
          return ts.factory.createTypeOperatorNode(ts.SyntaxKind.ReadonlyKeyword, child);
        }
        if (ts.isPropertySignature(child)) {
          let type = child.type;
          const owner = node.parent;
          if (
            owner &&
            ts.isInterfaceDeclaration(owner) &&
            child.name &&
            ts.isIdentifier(child.name)
          ) {
            const field = schema.$defs?.[owner.name.text]?.properties?.[child.name.text];
            const keyRef = field?.propertyNames?.$ref;
            const index =
              type && ts.isTypeLiteralNode(type)
                ? type.members.find(ts.isIndexSignatureDeclaration)
                : undefined;
            if (keyRef && index?.type) {
              const valueType = ts.isUnionTypeNode(index.type)
                ? ts.factory.createUnionTypeNode(
                    index.type.types.filter(
                      (member) => member.kind !== ts.SyntaxKind.UndefinedKeyword,
                    ),
                  )
                : index.type;
              type = ts.factory.createTypeReferenceNode("Readonly", [
                ts.factory.createTypeReferenceNode("Partial", [
                  ts.factory.createTypeReferenceNode("Record", [
                    ts.factory.createTypeReferenceNode(keyRef.split("/").at(-1)),
                    valueType,
                  ]),
                ]),
              ]);
            }
          }
          return ts.factory.updatePropertySignature(
            child,
            [ts.factory.createModifier(ts.SyntaxKind.ReadonlyKeyword)],
            child.name,
            child.questionToken,
            type,
          );
        }
        if (ts.isIndexSignatureDeclaration(child)) {
          return ts.factory.updateIndexSignature(
            child,
            [ts.factory.createModifier(ts.SyntaxKind.ReadonlyKeyword)],
            child.parameters,
            ts.factory.createUnionTypeNode([
              child.type,
              ts.factory.createKeywordTypeNode(ts.SyntaxKind.UndefinedKeyword),
            ]),
          );
        }
        return child;
      };
      return (node) => ts.visitNode(node, visit) as ts.SourceFile;
    },
  ]);
  const result = ts.createPrinter().printFile(transformed.transformed[0]);
  transformed.dispose();
  return result;
}

async function main() {
  const rawSchema = JSON.parse(readFileSync(SCHEMA_PATH, "utf-8"));
  const schema = intersectUnionProperties(stripFieldTitles(collapseRefs(rawSchema)));

  const ts = await compile(schema, "CausalSSMContracts", {
    bannerComment:
      "/* eslint-disable */\n" +
      "/**\n" +
      " * AUTO-GENERATED — DO NOT EDIT\n" +
      " *\n" +
      " * Generated from Python Pydantic models via:\n" +
      " *   cd apps/data-pipeline && uv run python -m scripts.codegen.export_api\n" +
      " *   cd packages/api-types && bun run scripts/generate.ts\n" +
      " *\n" +
      " * Source of truth: apps/data-pipeline/src/nof1_causal_lab/artifacts/catalog.py\n" +
      " * plus facade API models exported from apps/data-pipeline/src/nof1_causal_lab/study_api.py\n" +
      " */",
    additionalProperties: false,
    strictIndexSignatures: false,
    enableConstEnums: false,
    unreachableDefinitions: true,
    unknownAny: false,
    style: {
      semi: true,
      singleQuote: false,
    },
  });

  writeOrCheck(OUTPUT_PATH, readonlyOutputs(ts, rawSchema));

  // Count interfaces generated
  const count = (ts.match(/export (interface|type)/g) || []).length;
  if (!checkOnly) {
    console.log(`Generated ${count} types/interfaces → ${OUTPUT_PATH}`);
  }

  generateMetadata();

  if (checkOnly && changedPaths.length > 0) {
    console.error("TypeScript API type generation is out of date. Run `bun run codegen`.");
    for (const path of changedPaths) {
      console.error(`  ${path}`);
    }
    process.exit(1);
  }

  if (checkOnly) {
    console.log("TypeScript API type generation checked.");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
