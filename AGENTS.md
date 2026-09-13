- Add tests only when necessary to cover broad behavior; this repo's test suite is already costly.

- Project description: nof1-causal-lab is for observational longitudinal causal questions, especially intensive longitudinal data (ILD) and idiographic / N-of-1 settings where measurements are irregular, messy, and semantically heterogeneous. The LLM proposes constructs, indicators, causal structure, and priors. It combines explicit causal-identification checks with continuous-time latent state-space estimation, and only produces numeric causal claims when those checks support them.

- TODO references mean the gitignored `scratchpad/TODO.md`.

- `cp` means "commit and push". Split commits atomically, but prioritize shared-workspace safety: never disrupt another agent's work with stashing or rebasing.

- Never add backwards compatibility code or defensive fallbacks.

- Before integration testing, starting services, health checks, or manual pipeline runs, read and follow [docs/guides/agentic_integration_testing.md](docs/guides/agentic_integration_testing.md).

- Prefer `ast-grep` for code navigation.

- After substantive source changes, run `bun run duplicates` and review its advisory candidates. The audit is diff-aware; use `--deep` for a broader search and `--all` only for repository-wide audits.

# Notebooks

- `apps/data-pipeline/notebooks/` contains marimo Python notebooks.
- Name every top-level `@app.cell` function with a descriptive snake_case name. Keep inner helpers `_`-prefixed.
- From `apps/data-pipeline`, validate edits with `uv run marimo check --strict notebooks/<name>.py`.
- Launch the editor from the repo root with `bun run --cwd apps/data-pipeline notebooks`.
- For `marimo-pair`, open a specific notebook in the browser first; the file browser exposes no session. Attach with `execute-code.sh --url http://localhost:2718 --token <access_token>`, using the startup banner's token. Pass it via `--token` or `MARIMO_TOKEN`, never in the `--url` query string.

# Docs

- Place references beside the claims they support or hyperlink the relevant terms.
- After editing `README.md` or files under `docs/`, run `bun run docs:check`.

- In `docs/pipeline`, each stage doc owns its output and artifact definitions; downstream stages link to them.

- In `docs/pipeline`, Outputs sections use field/description tables for core artifacts, without extra dataclass prose. Omit internal plumbing (`outcome`, `llm_trace`) and wrappers such as `IndicatorAudit`.

# Web app

- Never put domain logic or statistical computations in frontend code.

- Reuse the dev server on port 3000 if running; ask before restarting it.
- Check errors with the next-devtools MCP.
- Use `bun` exclusively.

# Data Pipeline

- Budget GPU benchmarks carefully: a B200 on Modal costs $6/hour.

- Never run evals (`inspect eval`) unless explicitly asked. Use `uv run pytest tests/` for testing; avoid `slow` tests unless directly affected.

- Before committing, run `bun run --cwd apps/data-pipeline lint`.

- Represent structural assumptions as DAGs with explicit latent confounders. ADMGs are only for internal projection into y0's identification algorithm, never user-facing.

- The latent SSM is continuous-time **nonlinear**. Linearization and Gaussian approximations are allowed **only for particle-sampler initialization**: parameter positions, proposal preconditioner, and cSMC reference trajectory.
- Production posteriors, all diagnostics, posterior-predictive checks, and counterfactual/predictive outputs must use the exact engines: particle/SMC with the true emission density, Euler-Maruyama with the true nonlinear drift, and Diffrax for forward simulation.
- Exactly corrected proposals (`amala_exact`) are allowed. Uncorrected `amala` and `amala_plus` must remain non-default and never gate reported results.
- Before reintroducing linearization, run [test_linearization_init_only.py](apps/data-pipeline/tests/models/ssm/test_linearization_init_only.py), which restricts Laplace imports to warmup/init.
