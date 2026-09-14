/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python distribution catalog via:
 *   cd apps/data-pipeline && uv run python -m scripts.export_schemas
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/nof1_causal_lab/distributions.py
 */

import type { ArtifactId, ArtifactFileSpec, MachineDescription } from "./models";
export const MACHINE_DESCRIPTION: MachineDescription = {
  "artifact_ids": [
    "raw_data",
    "model",
    "identification_report",
    "panel",
    "validation_report"
  ],
  "topological_artifact_order": [
    "raw_data",
    "model",
    "identification_report",
    "panel",
    "validation_report"
  ],
  "topological_transition_order": [
    "raw_data",
    "latent_structure",
    "measurement_structure",
    "measurements",
    "statistical_model_spec",
    "posterior"
  ],
  "contexts": [
    {
      "context_id": "navigator",
      "layer": "navigator",
      "label": "Human/LLM navigator, web UI, SDK, or curl client",
      "parent_id": null,
      "owns": [],
      "allowed_tools": [],
      "runtime_state": []
    },
    {
      "context_id": "action-registry",
      "layer": "registry",
      "label": "Transport-independent action contracts",
      "parent_id": "navigator",
      "owns": [],
      "allowed_tools": [],
      "runtime_state": []
    },
    {
      "context_id": "episode-machine",
      "layer": "machine",
      "label": "Serialized artifact transition machine",
      "parent_id": "action-registry",
      "owns": [],
      "allowed_tools": [],
      "runtime_state": []
    },
    {
      "context_id": "ingestion",
      "layer": "delegated",
      "label": "Ingestion file/code loop",
      "parent_id": "episode-machine",
      "owns": [
        "raw_data"
      ],
      "allowed_tools": [
        "list_files",
        "read_file_sample",
        "execute_python",
        "submit_table"
      ],
      "runtime_state": [
        "prepared_input_dir",
        "sandbox",
        "result_df",
        "column_descriptions"
      ]
    },
    {
      "context_id": "latent-structure",
      "layer": "delegated",
      "label": "Latent structure proposal loop",
      "parent_id": "episode-machine",
      "owns": [
        "model"
      ],
      "allowed_tools": [
        "validate_latent_structure"
      ],
      "runtime_state": [
        "question",
        "latent_structure_draft"
      ]
    },
    {
      "context_id": "measurement-structure",
      "layer": "delegated",
      "label": "Measurement structure proposal loop",
      "parent_id": "episode-machine",
      "owns": [
        "model"
      ],
      "allowed_tools": [
        "validate_measurement_structure"
      ],
      "runtime_state": [
        "latent_structure",
        "dataset_schema",
        "measurement_structure_draft"
      ]
    },
    {
      "context_id": "measurement",
      "layer": "delegated",
      "label": "Indicator extraction worker fan-out",
      "parent_id": "episode-machine",
      "owns": [
        "panel"
      ],
      "allowed_tools": [
        "validate_extractions"
      ],
      "runtime_state": [
        "indicator_plan",
        "worker_statuses",
        "extracted_values"
      ]
    },
    {
      "context_id": "statistical-model-spec",
      "layer": "delegated",
      "label": "Model/prior reducer",
      "parent_id": "episode-machine",
      "owns": [
        "model"
      ],
      "allowed_tools": [
        "search_literature",
        "submit_construct"
      ],
      "runtime_state": [
        "deterministic_skeleton",
        "construct_order",
        "current_construct",
        "attempt",
        "checkpoint_ref",
        "accepted_constructs",
        "rebase"
      ]
    },
    {
      "context_id": "inference",
      "layer": "delegated",
      "label": "Exact nonlinear SSM inference job",
      "parent_id": "episode-machine",
      "owns": [
        "model"
      ],
      "allowed_tools": [],
      "runtime_state": [
        "sampler_config",
        "diagnostics",
        "conditioned_model"
      ]
    },
    {
      "context_id": "analysis",
      "layer": "tool",
      "label": "Runtime causal queries",
      "parent_id": "episode-machine",
      "owns": [],
      "allowed_tools": [
        "get_model_info",
        "simulate"
      ],
      "runtime_state": [
        "identified_treatments",
        "effect_summaries"
      ]
    }
  ],
  "actions": [
    {
      "action_id": "nav.state",
      "namespace": "nav",
      "name": "state",
      "kind": "read",
      "mode": "read",
      "context_id": "navigator",
      "consumes": [],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "nav.timeline",
      "namespace": "nav",
      "name": "timeline",
      "kind": "read",
      "mode": "read",
      "context_id": "navigator",
      "consumes": [],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "nav.events",
      "namespace": "nav",
      "name": "events",
      "kind": "read",
      "mode": "read",
      "context_id": "navigator",
      "consumes": [],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "nav.get",
      "namespace": "nav",
      "name": "get",
      "kind": "read",
      "mode": "read",
      "context_id": "navigator",
      "consumes": [],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "nav.versions",
      "namespace": "nav",
      "name": "versions",
      "kind": "read",
      "mode": "read",
      "context_id": "navigator",
      "consumes": [],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "nav.diff",
      "namespace": "nav",
      "name": "diff",
      "kind": "read",
      "mode": "read",
      "context_id": "navigator",
      "consumes": [],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "episode.create",
      "namespace": "episode",
      "name": "create",
      "kind": "produce",
      "mode": "direct",
      "context_id": "navigator",
      "consumes": [],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "derives": [],
      "move": {
        "kind": "write",
        "artifact_id": "model",
        "provenance": "human",
        "expected_model_version": 0
      },
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "episode.attach_data",
      "namespace": "episode",
      "name": "attach_data",
      "kind": "external",
      "mode": "direct",
      "context_id": "navigator",
      "consumes": [],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "episode.ingest_data",
      "namespace": "episode",
      "name": "ingest_data",
      "kind": "produce",
      "mode": "delegated",
      "context_id": "navigator",
      "consumes": [],
      "produces": [
        "raw_data"
      ],
      "produces_optional": [],
      "derives": [],
      "move": {
        "kind": "run",
        "operation_id": "raw_data"
      },
      "query": null,
      "lower_context_id": "ingestion"
    },
    {
      "action_id": "episode.refresh",
      "namespace": "episode",
      "name": "refresh",
      "kind": "driver",
      "mode": "async",
      "context_id": "navigator",
      "consumes": [],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "specify.latent_structure",
      "namespace": "specify",
      "name": "latent_structure",
      "kind": "produce",
      "mode": "delegated",
      "context_id": "navigator",
      "consumes": [
        "model"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "derives": [
        "identification_report",
        "validation_report"
      ],
      "move": {
        "kind": "run",
        "operation_id": "latent_structure"
      },
      "query": null,
      "lower_context_id": "latent-structure"
    },
    {
      "action_id": "specify.measurement",
      "namespace": "specify",
      "name": "measurement",
      "kind": "produce",
      "mode": "delegated",
      "context_id": "navigator",
      "consumes": [
        "raw_data",
        "model"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "derives": [
        "identification_report",
        "validation_report"
      ],
      "move": {
        "kind": "run",
        "operation_id": "measurement_structure"
      },
      "query": null,
      "lower_context_id": "measurement-structure"
    },
    {
      "action_id": "specify.edit",
      "namespace": "specify",
      "name": "edit",
      "kind": "produce",
      "mode": "direct",
      "context_id": "navigator",
      "consumes": [
        "model"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "derives": [
        "identification_report",
        "validation_report"
      ],
      "move": {
        "kind": "write",
        "artifact_id": "model",
        "provenance": "human",
        "expected_model_version": null
      },
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "specify.identify",
      "namespace": "specify",
      "name": "identify",
      "kind": "check",
      "mode": "direct",
      "context_id": "navigator",
      "consumes": [
        "model"
      ],
      "produces": [],
      "produces_optional": [],
      "derives": [
        "identification_report"
      ],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "measure.extract",
      "namespace": "measure",
      "name": "extract",
      "kind": "produce",
      "mode": "delegated",
      "context_id": "navigator",
      "consumes": [
        "raw_data",
        "model"
      ],
      "produces": [],
      "produces_optional": [
        "panel"
      ],
      "derives": [
        "validation_report"
      ],
      "move": {
        "kind": "run",
        "operation_id": "measurements"
      },
      "query": null,
      "lower_context_id": "measurement"
    },
    {
      "action_id": "fit.specify",
      "namespace": "fit",
      "name": "specify",
      "kind": "produce",
      "mode": "delegated",
      "context_id": "navigator",
      "consumes": [
        "model",
        "identification_report",
        "panel",
        "validation_report"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "derives": [
        "identification_report",
        "validation_report"
      ],
      "move": {
        "kind": "run",
        "operation_id": "statistical_model_spec"
      },
      "query": null,
      "lower_context_id": "statistical-model-spec"
    },
    {
      "action_id": "fit.infer",
      "namespace": "fit",
      "name": "infer",
      "kind": "produce",
      "mode": "async",
      "context_id": "navigator",
      "consumes": [
        "model",
        "panel"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "derives": [
        "identification_report",
        "validation_report"
      ],
      "move": {
        "kind": "run",
        "operation_id": "posterior"
      },
      "query": null,
      "lower_context_id": "inference"
    },
    {
      "action_id": "fit.check",
      "namespace": "fit",
      "name": "check",
      "kind": "check",
      "mode": "direct",
      "context_id": "navigator",
      "consumes": [
        "model"
      ],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    },
    {
      "action_id": "analyze.simulate",
      "namespace": "analyze",
      "name": "simulate",
      "kind": "query",
      "mode": "direct",
      "context_id": "navigator",
      "consumes": [
        "model",
        "panel",
        "identification_report"
      ],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": {
        "context_id": "analysis",
        "tool_name": "simulate",
        "freshness_checked": true
      },
      "lower_context_id": null
    },
    {
      "action_id": "analyze.counterfactual",
      "namespace": "analyze",
      "name": "counterfactual",
      "kind": "query",
      "mode": "direct",
      "context_id": "navigator",
      "consumes": [
        "model",
        "panel",
        "identification_report"
      ],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": {
        "context_id": "analysis",
        "tool_name": "simulate",
        "freshness_checked": true
      },
      "lower_context_id": null
    },
    {
      "action_id": "analyze.ppc",
      "namespace": "analyze",
      "name": "ppc",
      "kind": "check",
      "mode": "direct",
      "context_id": "navigator",
      "consumes": [
        "model",
        "panel"
      ],
      "produces": [],
      "produces_optional": [],
      "derives": [],
      "move": null,
      "query": null,
      "lower_context_id": null
    }
  ],
  "roots": [
    {
      "artifact_id": "model",
      "write_pins": []
    }
  ],
  "transitions": [
    {
      "transition_id": "raw_data",
      "consumes": [],
      "produces": [
        "raw_data"
      ],
      "produces_optional": [],
      "creation_class": "batch_llm",
      "writable": false
    },
    {
      "transition_id": "latent_structure",
      "consumes": [
        "model"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "creation_class": "judgment",
      "writable": false
    },
    {
      "transition_id": "measurement_structure",
      "consumes": [
        "raw_data",
        "model"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "creation_class": "judgment",
      "writable": false
    },
    {
      "transition_id": "measurements",
      "consumes": [
        "raw_data",
        "model"
      ],
      "produces": [],
      "produces_optional": [
        "panel"
      ],
      "creation_class": "batch_llm",
      "writable": false
    },
    {
      "transition_id": "statistical_model_spec",
      "consumes": [
        "model",
        "identification_report",
        "panel",
        "validation_report"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "creation_class": "judgment",
      "writable": false
    },
    {
      "transition_id": "posterior",
      "consumes": [
        "model",
        "panel"
      ],
      "produces": [
        "model"
      ],
      "produces_optional": [],
      "creation_class": "deterministic",
      "writable": false
    }
  ],
  "derivations": [
    {
      "produces": "identification_report",
      "from": [
        "model"
      ],
      "optional": false
    },
    {
      "produces": "validation_report",
      "from": [
        "panel",
        "model"
      ],
      "optional": false
    }
  ],
  "files": {
    "raw_data": {
      "json": {},
      "parquet": {
        "raw": "raw.parquet"
      }
    },
    "model": {
      "json": {
        "model": "model.json"
      },
      "parquet": {}
    },
    "identification_report": {
      "json": {
        "identification_report": "identification_report.json"
      },
      "parquet": {}
    },
    "panel": {
      "json": {},
      "parquet": {
        "panel": "panel.parquet"
      }
    },
    "validation_report": {
      "json": {
        "validation_report": "validation_report.json"
      },
      "parquet": {}
    }
  }
};
export const ARTIFACT_IDS = ["raw_data","model","identification_report","panel","validation_report"] as const satisfies readonly ArtifactId[];
export const ARTIFACT_FILE_SPECS: Record<ArtifactId, ArtifactFileSpec> = {
  "raw_data": {
    "json": {},
    "parquet": {
      "raw": "raw.parquet"
    }
  },
  "model": {
    "json": {
      "model": "model.json"
    },
    "parquet": {}
  },
  "identification_report": {
    "json": {
      "identification_report": "identification_report.json"
    },
    "parquet": {}
  },
  "panel": {
    "json": {},
    "parquet": {
      "panel": "panel.parquet"
    }
  },
  "validation_report": {
    "json": {
      "validation_report": "validation_report.json"
    },
    "parquet": {}
  }
};

const _OBS_HYPERS_BY_DIST = {
  "student_t": [
    "obs_df"
  ],
  "gamma": [
    "obs_shape"
  ],
  "negative_binomial": [
    "obs_r"
  ],
  "beta": [
    "obs_concentration"
  ],
  "ordered_logistic": [
    "obs_ordered_base",
    "obs_ordered_gaps"
  ],
  "categorical": [
    "obs_cat_intercepts",
    "obs_cat_slopes"
  ]
} as const;

export type ObservationHyperparameter =
  typeof _OBS_HYPERS_BY_DIST[keyof typeof _OBS_HYPERS_BY_DIST][number];

export const OBSERVATION_HYPERPARAMETERS_BY_DISTRIBUTION: Partial<
  Record<string, readonly ObservationHyperparameter[]>
> = _OBS_HYPERS_BY_DIST;
