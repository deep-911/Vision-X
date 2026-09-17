# Project Rules

- **Architecture**: Do NOT reduce or redesign the architecture from `SIH26038_Idea.md`.
- **MVP vs Final**: Implement MVP as the first validated vertical slice in `mvp/`, then evolve `final_project/` incrementally from it.
- **Datasets**: Validate IDRiD extraction and manifest first. Do not download other datasets (APTOS, EyePACS, Messidor-2) until the IDRiD pipeline is validated.
- **Artifacts**: Store large generated files in `artifacts/checkpoints/`, `artifacts/exports/`, `artifacts/metrics/`, `artifacts/reports/` and keep them out of Git.
