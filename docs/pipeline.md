# Scientific Actions and Optional Pipeline

The primary interface has four [scientific actions](reference/scientific-actions.md):
`edit_model`, `prepare_data`, `fit`, and `simulate`. Model editing can interleave
structure, measurements, parameters and laws. Applicable cheap checks refresh on
submission; fitting and simulation are explicit computations.

The table describes jobs and derivations used by the optional observational-study
recipe. Its ordering is a navigation policy, separate from direct action readiness.
Each artifact's definition lives in the doc that introduces it. See
[pipeline dimensions](reference/pipeline-dimensions.md) for shared concerns.

| Run artifact / derived view | Name | Primary artifact | Modality | Interactive | Stop condition | File |
|---|---|---|---|---|---|---|
| `raw_data` | Agentic Data Ingestion | [`Raw dataframe`](pipeline/ingestion.md#raw-dataframe) | Semantic | No | None | [pipeline/ingestion.md](pipeline/ingestion.md) |
| `latent_structure` | Latent Structure Proposal | `ModelSpec` | Semantic | Yes | None | [pipeline/latent-structure.md](pipeline/latent-structure.md) |
| `measurement_structure` | Measurement Structure and Identifiability | `ModelSpec` | Semantic | Yes | Stops if no identifiable treatments remain | [pipeline/measurement-structure.md](pipeline/measurement-structure.md) |
| `measurements` | Indicator Extraction | `ObservationRecord`s | Hybrid | No | Stops if no `ObservationRecord`s are extracted | [pipeline/extraction.md](pipeline/extraction.md) |
| `validation_report` | Extraction Validation | Indicator audits | Computed | No | Stops on validation errors | [pipeline/extraction-validation.md](pipeline/extraction-validation.md) |
| `statistical_model_spec` | Statistical Model Specification and Prior Elicitation | `ModelSpec` + priors | Semantic | Yes | None | [pipeline/statistical-model-spec.md](pipeline/statistical-model-spec.md) |
| `posterior` | Inference and Diagnostics | Conditioned `ModelSpec` + fit diagnostics | Computed | No | Stops if model fitting fails | [pipeline/inference.md](pipeline/inference.md) |
| `data_profile` | Data Profile | Model-independent empirical findings | Computed | No | Reports findings | [pipeline/extraction-validation.md](pipeline/extraction-validation.md#independent-data-profiles) |
| `simulate` | Simulation | Arrays, measurements and optional causal result | Computed | No | Causal results require identification and production-fit evidence | [reference/scientific-actions.md](reference/scientific-actions.md#simulation-report) |

Ordinary simulation can run before or after fitting by sampling the selected model's
current laws. [Causal simulation](pipeline/analysis.md) uses the same action with
additional evidence requirements. Its results persist in the journal.
