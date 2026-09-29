# ASTRA Vision

Project author: Preetham Alawandimath.

ASTRA Vision analyzes defence imagery within a transparent five-class scope and attaches credited catalog evidence when a supplied reference supports an exact or visual match.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- The API reads the supplied starter dataset from `artifacts/astra-vision/public/dataset`; no database is required for the current review-only history.

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `artifacts/astra-vision` — React/Vite review desk with live analysis, evaluation, catalog, history, and methodology routes.
- `artifacts/api-server/src/services/vision.ts` — catalog loading, exact/visual evidence matching, zero-shot classification, and response assembly.
- `artifacts/api-server/src/routes/vision.ts` — model, evaluation, catalog, history, and multipart analysis endpoints.
- `lib/api-spec/openapi.yaml` — source of truth for the generated API client and Zod response schemas.
- `artifacts/astra-vision/public/dataset` — 150 supplied images plus `labels.csv` and `credits.csv`.

## Architecture decisions

- The starter dataset has image-level labels only, so the app does not claim object detection or bounding boxes.
- Exact vehicle names are evidence-backed catalog results, not inferred class labels.
- The classifier is a five-class zero-shot baseline; evaluation metrics remain `Not evaluated` until a held-out fine-tuned checkpoint is recorded.
- Reference images and Wikimedia credit fields are kept together so every catalog result can be audited.

## Product

Users can upload JPG, PNG, or WEBP images, inspect supported-class predictions and top alternatives, review exact or visual-neighbor reference evidence, browse the credited catalog, and see honest dataset/evaluation notes.

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- The API workflow runs from `artifacts/api-server`, so the vision service resolves the dataset from sibling artifact paths.
- Do not convert the supplied catalog reference into a universal accuracy claim; the UI intentionally exposes uncertainty and evaluation state.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
