# ASTRA Vision — Vehicle Recognition & Evidence Review

> **Catalog-aware computer-vision review application for bounded five-class image classification, reference matching, and traceable catalog evidence.**

ASTRA Vision is a full-stack TypeScript/React computer-vision review prototype. It accepts an image, validates and normalizes it, performs zero-shot image classification with **`Xenova/clip-vit-base-patch32`**, checks the supplied reference catalog for an **exact SHA-256 match**, and can surface a **visual-neighbor reference** when the configured similarity threshold is met.

The application is intentionally scoped to five high-level classes:

- Aircraft
- Helicopter
- Drone
- Military Vehicle
- Naval Vessel

The project is designed as a transparent review/recognition prototype. It **does not claim object detection, bounding-box localization, real-world identity verification, operational intelligence, or validated production-level classification accuracy**.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Key Features](#2-key-features)
- [3. Technology Stack](#3-technology-stack)
- [4. System Architecture](#4-system-architecture)
- [5. Request and Analysis Flow](#5-request-and-analysis-flow)
- [6. Repository Structure](#6-repository-structure)
- [7. Prerequisites](#7-prerequisites)
- [8. Installation](#8-installation)
- [9. Environment Variables](#9-environment-variables)
- [10. Running the Project](#10-running-the-project)
- [11. Verification and Build](#11-verification-and-build)
- [12. Using the Application](#12-using-the-application)
- [13. API Reference](#13-api-reference)
- [14. Dataset](#14-dataset)
- [15. Evidence and Matching Model](#15-evidence-and-matching-model)
- [16. Model and Classification](#16-model-and-classification)
- [17. Evaluation Status](#17-evaluation-status)
- [18. Technical Design Decisions](#18-technical-design-decisions)
- [19. Security and File Handling](#19-security-and-file-handling)
- [20. Known Limitations](#20-known-limitations)
- [21. Troubleshooting](#21-troubleshooting)
- [22. Development and API Contract](#22-development-and-api-contract)
- [23. Responsible Use](#23-responsible-use)
- [24. Submission Checklist](#24-submission-checklist)
- [25. License and Attribution](#25-license-and-attribution)
- [26. Project Status](#26-project-status)

---

## 1. Project Overview

ASTRA Vision provides an end-to-end workflow for reviewing an uploaded image against a bounded five-class vocabulary and the supplied reference catalog.

### What happens when an image is uploaded?

```text
User uploads image
        |
        v
File type / size / decodability validation
        |
        v
Image normalization and metadata extraction
        |
        +-----------------------------+
        |                             |
        v                             v
SHA-256 exact-file lookup       Zero-shot CLIP classification
        |                             |
        |                             v
        |                       Class + alternatives
        |                             |
        +-------------+---------------+
                      |
                      v
             Visual-neighbor lookup
                      |
                      v
          Evidence-aware API response
                      |
                      v
              React review interface
```

The result is designed to show:

- predicted class
- confidence signal
- alternative classes
- evidence state
- catalog reference information when supported
- image metadata
- preprocessing information
- model mode
- inference timing
- limitations/evaluation state

---

## 2. Key Features

### Image Analysis

- Upload **JPG/JPEG, PNG, and WEBP** images.
- Maximum upload size: **10 MB**.
- Server-side validation before analysis.
- Image normalization and metadata extraction using **Sharp**.

### Five-Class Zero-Shot Classification

The classifier uses:

```text
Xenova/clip-vit-base-patch32
```

through Hugging Face Transformers / ONNX Runtime.

Supported labels:

```text
Aircraft
Helicopter
Drone
Military Vehicle
Naval Vessel
```

The interface also exposes alternative predictions rather than presenting only one label.

### Exact Reference Matching

The uploaded file is checked against the supplied catalog using **SHA-256**.

An exact match means the uploaded bytes correspond to a catalog image.

### Visual-Neighbor Matching

When there is no exact file match, ASTRA Vision can compare the image against the supplied catalog using normalized grayscale image signatures and cosine similarity.

A visual neighbor is presented as a **reference**, not as proof that the uploaded image contains the same real-world vehicle.

### Catalog Evidence

The supplied catalog contains:

- image labels
- reference images
- source title
- creator
- license
- source page / Wikimedia Commons record

### Review UI

The frontend provides pages/features for:

- Live Analysis
- Evaluation / methodology
- Reference Catalog
- Analysis History
- About / project information

### API

The project exposes endpoints for:

- health
- model information
- evaluation status
- catalog
- analysis history
- image analysis

---

## 3. Technology Stack

| Layer | Technology |
|---|---|
| Frontend | React 19 |
| Frontend build | Vite 7 |
| Language | TypeScript 5.9 |
| Backend | Express 5 |
| ML inference | Hugging Face Transformers |
| ML runtime | ONNX Runtime Node |
| Image processing | Sharp |
| File upload | Multer |
| Validation | Zod |
| Data fetching | TanStack Query |
| Styling | Tailwind CSS |
| UI components | Radix UI |
| Logging | Pino / Pino HTTP |
| API contract | OpenAPI |
| Package manager | pnpm |
| Workspace | pnpm workspace |

The exact dependency versions and transitive dependency graph are pinned by the repository's package manifests and `pnpm-lock.yaml`.

There is **no Python `requirements.txt`** because ASTRA Vision is a Node.js/TypeScript application.

---

## 4. System Architecture

```text
                         ┌──────────────────────────┐
                         │      React / Vite UI      │
                         │                          │
                         │ Live Analysis             │
                         │ Evaluation               │
                         │ Catalog                  │
                         │ History                  │
                         │ About                    │
                         └────────────┬─────────────┘
                                      │
                              HTTP / JSON /
                           multipart form-data
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       Express API        │
                         │                          │
                         │ GET  /api/health         │
                         │ GET  /api/vision/*       │
                         │ POST /api/vision/analyze │
                         └────────────┬─────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
                  ▼                   ▼                   ▼
          ┌──────────────┐   ┌────────────────┐   ┌────────────────┐
          │ Image        │   │ Reference      │   │ CLIP           │
          │ Validation   │   │ Catalog        │   │ Zero-Shot      │
          │ + Sharp      │   │ labels.csv     │   │ Classification  │
          │ + Multer     │   │ credits.csv    │   │ ONNX Runtime   │
          └──────┬───────┘   │ 150 images     │   └───────┬────────┘
                 │            └───────┬────────┘           │
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      ▼
                         ┌──────────────────────────┐
                         │ Evidence-Aware Response  │
                         │                          │
                         │ Class                    │
                         │ Confidence signal        │
                         │ Alternatives             │
                         │ Evidence state           │
                         │ Catalog reference        │
                         │ Metadata                 │
                         │ Timing                   │
                         │ Limitations              │
                         └──────────────────────────┘
```

### Core components

#### Frontend

Located under:

```text
artifacts/astra-vision/
```

Responsible for:

- image upload
- analysis UI
- results presentation
- catalog browsing
- evaluation/methodology views
- analysis history
- API interaction

#### API Server

Located under:

```text
artifacts/api-server/
```

Responsible for:

- HTTP API
- upload validation
- image processing
- model inference
- catalog matching
- analysis history
- structured responses

#### API Contract

Located under:

```text
lib/api-spec/
lib/api-zod/
lib/api-client-react/
```

The OpenAPI specification is the API contract source of truth, with generated schemas/client support.

#### Reference Dataset

Located under:

```text
artifacts/astra-vision/public/dataset/
```

---

## 5. Request and Analysis Flow

The main endpoint is:

```http
POST /api/vision/analyze
```

with:

```text
Content-Type: multipart/form-data
```

and:

```text
image=<JPG|PNG|WEBP file>
```

### Processing sequence

1. The frontend sends the selected image to the API.
2. The API validates the uploaded file.
3. The image size and supported MIME type are checked.
4. Sharp decodes/normalizes the image and extracts metadata.
5. ASTRA Vision calculates/checks the file's SHA-256 hash against the reference catalog.
6. If an exact match is found, the corresponding catalog evidence is attached.
7. If no exact match is found, a visual-neighbor signature can be calculated.
8. The CLIP zero-shot classifier produces class scores.
9. Alternative predictions are retained.
10. A catalog neighbor can be attached only when the configured evidence threshold is satisfied.
11. The API returns a structured, evidence-aware response.
12. The frontend renders the result and associated evidence.

---

## 6. Repository Structure

```text
.
├── artifacts/
│   ├── api-server/
│   │   └── # Express API and vision service
│   │
│   ├── astra-vision/
│   │   └── # React/Vite frontend and bundled dataset
│   │
│   └── mockup-sandbox/
│       └── # UI sandbox artifact
│
├── lib/
│   ├── api-client-react/
│   │   └── # Generated API client
│   │
│   ├── api-spec/
│   │   └── # OpenAPI source of truth
│   │
│   ├── api-zod/
│   │   └── # Generated Zod schemas
│   │
│   └── db/
│       └── # Drizzle/PostgreSQL package retained for workspace compatibility
│
├── scripts/
│   └── # Workspace scripts
│
├── docs/
│   ├── REQUIREMENTS.md
│   ├── DATASET.md
│   ├── ARCHITECTURE.md
│   ├── VIDEO_SCRIPT.md
│   └── SUBMISSION_CHECKLIST.md
│
├── attached_assets/
│   └── # Project assets
│
├── package.json
├── pnpm-workspace.yaml
├── pnpm-lock.yaml
├── tsconfig.json
├── .env.example
└── README.md
```

---

## 7. Prerequisites

### Required software

Install the following before starting:

- **Node.js 24.x**
- **pnpm**
- **Git**

The application performs CPU-based ONNX inference, so use a machine with enough memory and processing capacity for the CLIP model.

### Verify installations

```bash
node --version
pnpm --version
git --version
```

The project was designed for Node 24.x.

If your Node version is different, use Node Version Manager (`nvm`) or another Node version manager to switch to Node 24.

---

## 8. Installation

### Step 1 — Clone the repository

Replace the placeholder with your actual GitHub repository URL:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd ASTRA-Vision-Vehicle-Recognition
```

Example:

```bash
git clone https://github.com/YOUR_USERNAME/ASTRA-Vision-Vehicle-Recognition.git
cd ASTRA-Vision-Vehicle-Recognition
```

### Step 2 — Confirm the branch

```bash
git branch
```

Make sure you are on the branch containing the final submission code.

### Step 3 — Install dependencies

```bash
pnpm install
```

The repository intentionally uses pnpm through the root package configuration.

### Step 4 — Configure environment variables

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then review `.env`.

---

## 9. Environment Variables

The project uses environment variables for server/runtime configuration.

### Example `.env`

```env
PORT=5000
BASE_PATH=/
NODE_ENV=development
LOG_LEVEL=info
```

### Variable reference

| Variable | Required | Example | Description |
|---|---:|---|---|
| `PORT` | Yes | `5000` | Port used by the active server/process. |
| `BASE_PATH` | Yes for Vite app | `/` | Vite base path used when serving the frontend. |
| `NODE_ENV` | No | `development` | Runtime environment. |
| `LOG_LEVEL` | No | `info` | Pino logging level. |

### Important

Do **not** commit:

```text
.env
```

to GitHub.

Commit:

```text
.env.example
```

instead.

The configured Hugging Face model does not require a private model API key.

If the model is not already cached locally, the first model load may require internet access to obtain the model files.

---

## 10. Running the Project

ASTRA Vision is a workspace application containing the API server and frontend.

### Option A — Run the API server

From the repository root:

```bash
pnpm --filter @workspace/api-server run dev
```

The API server expects `PORT` to be configured.

### Option B — Run the frontend

In another terminal:

```bash
pnpm --filter @workspace/astra-vision run dev
```

The Vite application expects the appropriate `PORT` and `BASE_PATH` configuration.

### Recommended local workflow

Use two terminals.

**Terminal 1 — API**

```bash
pnpm --filter @workspace/api-server run dev
```

**Terminal 2 — Frontend**

```bash
pnpm --filter @workspace/astra-vision run dev
```

Then open the local URL printed by the Vite development server.

> The exact frontend/API port behavior depends on the workspace runtime configuration. Use the URLs printed in the terminal rather than assuming a fixed browser port.

---

## 11. Verification and Build

Before submitting the project, run the following checks.

### Type checking

```bash
pnpm run typecheck
```

Expected result:

```text
No TypeScript errors
```

### Production build

```bash
pnpm run build
```

A successful build confirms that the workspace can be compiled for deployment/review.

### Recommended final validation

Run:

```bash
pnpm install
pnpm run typecheck
pnpm run build
```

Then start the application and perform at least one complete image-analysis test through the UI.

---

## 12. Using the Application

### Basic workflow

1. Start the API server.
2. Start the frontend.
3. Open the frontend in your browser.
4. Navigate to **Live Analysis**.
5. Upload a supported image.
6. Wait for analysis to complete.
7. Review:
   - predicted class
   - confidence signal
   - alternative predictions
   - evidence status
   - catalog reference, if available
   - metadata
   - inference timing
8. Use the **Catalog** section to inspect supplied reference images and attribution.
9. Use **Evaluation** to inspect methodology and evaluation status.
10. Use **History** to review recent analysis activity.

### Supported upload formats

```text
.jpg
.jpeg
.png
.webp
```

Maximum upload size:

```text
10 MB
```

---

## 13. API Reference

The API is available under `/api`.

### Health check

```http
GET /api/health
```

Use this to verify that the backend is running.

### Model information

```http
GET /api/vision/model-info
```

Returns model/mode information used by the vision service.

### Evaluation report

```http
GET /api/vision/evaluation
```

Returns the project's current evaluation/methodology state.

### Reference catalog

```http
GET /api/vision/catalog
```

Returns catalog information used by the review workflow.

### Analysis history

```http
GET /api/vision/history
```

Returns recent in-memory analysis history.

### Analyze an image

```http
POST /api/vision/analyze
Content-Type: multipart/form-data
```

Multipart field:

```text
image=<JPG|PNG|WEBP file>
```

Example with `curl`:

```bash
curl -X POST \
  http://localhost:5000/api/vision/analyze \
  -F "image=@/path/to/image.jpg"
```

> Replace `5000` with the actual API port configured for your local environment.

### API contract

The OpenAPI source is maintained under:

```text
lib/api-spec/
```

Generated API schemas/client code are maintained under:

```text
lib/api-zod/
lib/api-client-react/
```

---

## 14. Dataset

The supplied dataset is located at:

```text
artifacts/astra-vision/public/dataset/
```

### Dataset contents

```text
dataset/
├── labels.csv
├── credits.csv
└── images/
    ├── ...
    └── ...
```

### `labels.csv`

Maps supplied images to their supported classes.

### `credits.csv`

Contains reference metadata such as:

- source title
- creator
- license
- source page / Wikimedia Commons record

### `images/`

Contains the supplied reference image set.

The current dataset contains:

```text
150 reference images
```

across the five supported classes.

### Important dataset principle

The application reads the supplied catalog as part of the review workflow. The catalog is not treated as a general-purpose representation of all possible aircraft, helicopters, drones, military vehicles, or naval vessels.

See:

```text
docs/DATASET.md
```

for the project's dataset protocol and attribution details.

---

## 15. Evidence and Matching Model

ASTRA Vision separates classification from reference evidence.

### Evidence state 1 — Exact file match

```text
exact-file-match
```

The uploaded file has the same SHA-256 hash as a catalog reference.

This provides deterministic evidence that the uploaded bytes correspond to that catalog file.

### Evidence state 2 — Visual neighbor

```text
visual-neighbor
```

The uploaded image is sufficiently similar to a credited catalog reference under the configured visual-signature method.

This is a **reference signal**, not proof that the uploaded image depicts the same real-world vehicle.

### Evidence state 3 — Catalog reference

```text
catalog-reference
```

Reserved for catalog-backed reference representation.

### Why the distinction matters

A visually similar image can differ in:

- viewpoint
- scale
- crop
- lighting
- background
- camera quality
- vehicle variant
- image source

Therefore, ASTRA Vision does not turn visual similarity into an unsupported identity claim.

---

## 16. Model and Classification

### Model

```text
Xenova/clip-vit-base-patch32
```

The model is used through Hugging Face Transformers with ONNX Runtime.

### Classification approach

ASTRA Vision uses a zero-shot image/text classification baseline.

The supported vocabulary is deliberately limited to:

```text
Aircraft
Helicopter
Drone
Military Vehicle
Naval Vessel
```

This keeps the model output aligned with the labels represented by the supplied starter dataset.

### Why zero-shot CLIP?

The project requires a functional computer-vision baseline without claiming that a custom production model has been trained.

CLIP provides a practical image/text zero-shot approach while allowing the supported vocabulary to remain explicit.

### Confidence interpretation

Model scores are **confidence signals** from the zero-shot classification pipeline.

They should not be interpreted as:

- validated probability of correctness
- benchmark accuracy
- real-world identification certainty
- operational intelligence

---

## 17. Evaluation Status

### Current status

The application deliberately reports:

```text
Not evaluated
```

for metrics such as:

- accuracy
- precision
- recall
- F1
- top-3 accuracy

### Why?

The repository contains:

- a labeled starter dataset
- a working zero-shot baseline
- an evaluation/methodology interface

However, the current project does **not** contain a recorded held-out evaluation run or validated fine-tuned checkpoint from which those metrics can honestly be reported.

Therefore, the project does not fabricate performance numbers.

### Planned evaluation protocol

The UI/documentation describes a future stratified:

```text
80% training
10% validation
10% test
```

style split as an evaluation plan.

This should be understood as a **planned methodology**, not measured performance.

---

## 18. Technical Design Decisions

### Why React + Vite?

React/Vite provides a fast browser-based review interface with a straightforward development/build workflow.

### Why Express?

Express provides a lightweight API layer for:

- upload handling
- image analysis
- model inference
- catalog operations
- history

### Why TypeScript?

TypeScript provides compile-time checks across the frontend, backend, API schemas, and shared workspace packages.

### Why Sharp?

Sharp is used for image decoding/normalization and metadata processing.

### Why exact SHA-256 matching?

SHA-256 gives deterministic file identity.

If two files have the same hash, the application can establish that the uploaded bytes correspond to the catalog file.

This is much more auditable than treating a visual similarity score as exact identity.

### Why visual-neighbor matching?

When an uploaded image is not an exact catalog file, the system can still surface a potentially useful credited visual reference.

The UI keeps this as a reference rather than presenting it as proof.

### Why five classes?

The supplied starter labels support five high-level categories. Keeping the classifier vocabulary aligned with the available labels avoids implying unsupported fine-grained recognition.

### Why no bounding boxes?

The supplied labels are image-level labels.

Without bounding-box annotations, the application does not claim object localization.

---

## 19. Security and File Handling

ASTRA Vision performs server-side checks on uploaded images.

### Upload controls

The application supports:

```text
JPG/JPEG
PNG
WEBP
```

and limits uploads to:

```text
10 MB
```

The API also validates that the uploaded image can be decoded before analysis.

### Environment security

Never commit secrets:

```text
.env
```

Use:

```text
.env.example
```

for documented configuration.

### Logging

The project uses Pino/Pino HTTP logging.

Review logs carefully before publishing them if your deployment environment contains sensitive information.

### History storage

The current analysis history is lightweight and in-memory. It is intended for review/demo use and is not a durable production audit database.

---

## 20. Known Limitations

The current implementation has several important limitations.

### Dataset size

The supplied dataset contains 150 reference images. This is relatively small for general-purpose computer vision.

### Image-level labels

The dataset does not provide bounding-box annotations, so the application does not support validated object detection.

### Zero-shot baseline

CLIP zero-shot scores are not equivalent to validated classification accuracy.

### Visual-neighbor matching

Visual similarity does not prove real-world identity.

### Evaluation

Formal classification metrics are currently unmeasured.

### Hardware

CPU inference can be slower than GPU inference.

### History

The history store is in-memory and is intended for review/demo usage rather than durable production auditing.

### Dataset licensing

Reference images and attribution information must remain associated with redistributed material according to their applicable licenses.

---

## 21. Troubleshooting

### `pnpm: command not found`

Install pnpm and verify:

```bash
pnpm --version
```

If you use Corepack with your Node installation, you can enable it with:

```bash
corepack enable
```

Then retry:

```bash
pnpm install
```

### Wrong Node version

Check:

```bash
node --version
```

The project was designed for Node 24.x.

Use a Node version manager such as `nvm` if you need to switch versions.

### Dependencies fail to install

From the repository root:

```bash
rm -rf node_modules
pnpm install
```

On Windows PowerShell:

```powershell
Remove-Item -Recurse -Force node_modules
pnpm install
```

Do not delete `pnpm-lock.yaml` unless you intentionally want to regenerate the dependency lockfile.

### TypeScript errors

Run:

```bash
pnpm run typecheck
```

Check the first reported error before addressing downstream errors.

### Build fails

Run:

```bash
pnpm run build
```

Then verify:

- Node version
- pnpm version
- `.env`
- dependency installation
- workspace package names
- generated API artifacts

### API does not respond

Check whether the API process is running:

```bash
pnpm --filter @workspace/api-server run dev
```

Then test:

```bash
curl http://localhost:5000/api/health
```

Replace `5000` with your configured API port if necessary.

### Frontend cannot reach API

Verify:

1. The API server is running.
2. The API is listening on the expected port.
3. The frontend is running.
4. Your environment configuration matches the workspace setup.
5. The generated API client matches the current OpenAPI contract.

### First model load is slow

The first model load can take longer if model files are not already cached.

Internet access may be required during the initial model download.

Subsequent runs can use the locally available model cache.

---

## 22. Development and API Contract

### Type checking

```bash
pnpm run typecheck
```

### Build

```bash
pnpm run build
```

### OpenAPI code generation

If the API contract changes, update:

```text
lib/api-spec/openapi.yaml
```

Then regenerate the related artifacts:

```bash
pnpm --filter @workspace/api-spec run codegen
```

Do not manually edit generated API artifacts when they are expected to be regenerated from the OpenAPI source.

### Recommended development order

When changing an API:

1. Update OpenAPI specification.
2. Regenerate schemas/client.
3. Update server implementation.
4. Update frontend usage.
5. Run type checking.
6. Run production build.
7. Test the affected endpoint in the UI.

---

## 23. Responsible Use

ASTRA Vision is a transparent image-review prototype.

Its output should be independently reviewed and should **not** be treated as proof of:

- real-world identity
- ownership
- intent
- operational status
- location
- affiliation
- other consequential facts

Model confidence and visual similarity are signals produced by the implemented methods, not authoritative conclusions.

The project intentionally communicates its limitations rather than presenting unmeasured performance as fact.

---

## 24. Submission Checklist

Before submitting the GitHub repository, verify:

### Repository

- [ ] Source code is present.
- [ ] `README.md` is present and complete.
- [ ] `package.json` is present.
- [ ] `pnpm-lock.yaml` is committed.
- [ ] `pnpm-workspace.yaml` is committed.
- [ ] `.env.example` is committed.
- [ ] Real `.env` is **not** committed.
- [ ] Relevant configuration files are committed.
- [ ] Dataset instructions are included.
- [ ] Dataset attribution/credits are included.
- [ ] `docs/` contains the supporting submission documents.

### Local verification

Run:

```bash
pnpm install
pnpm run typecheck
pnpm run build
```

Then launch the application and test:

- [ ] Image upload works.
- [ ] Valid JPG/JPEG/PNG/WEBP image is accepted.
- [ ] Oversized/unsupported files are rejected appropriately.
- [ ] Analysis returns a result.
- [ ] Alternative predictions are shown.
- [ ] Catalog/evidence information renders correctly.
- [ ] Evaluation page loads.
- [ ] History page loads.
- [ ] No secret values are exposed.

### Video

Record a **5–8 minute technical project explanation** covering:

1. Problem statement
2. Project scope
3. Architecture
4. Frontend
5. Backend/API
6. Image-processing pipeline
7. CLIP zero-shot model
8. Exact SHA-256 matching
9. Visual-neighbor evidence
10. Dataset
11. Evaluation status
12. Technical decisions
13. Limitations
14. Live demonstration

Use:

```text
docs/VIDEO_SCRIPT.md
```

as the recording guide.

---

## 25. License and Attribution

The application source code and the supplied reference dataset are separate concerns.

Before redistributing reference images, preserve the attribution and licensing information supplied in:

```text
artifacts/astra-vision/public/dataset/credits.csv
```

and:

```text
docs/DATASET.md
```

Do not remove creator, license, or source information from credited reference material.

For the exact licensing terms of individual reference images, consult their corresponding source records.

---

## 26. Project Status

### Current implementation

| Component | Status |
|---|---|
| React/Vite frontend | Implemented |
| Express API | Implemented |
| Image upload | Implemented |
| Image validation | Implemented |
| Sharp preprocessing | Implemented |
| CLIP zero-shot classification | Implemented |
| Five-class vocabulary | Implemented |
| SHA-256 exact matching | Implemented |
| Visual-neighbor matching | Implemented |
| Catalog evidence | Implemented |
| Analysis history | Implemented |
| OpenAPI contract | Implemented |
| Dataset/credits | Included |
| Formal benchmark metrics | Not yet measured |
| Bounding-box detection | Not supported |

### Project maturity

ASTRA Vision should currently be understood as a **working research/demo/review prototype**, not a production-grade intelligence or object-detection platform.

The implementation intentionally favors:

- bounded scope
- transparent evidence
- reproducibility
- explicit limitations
- traceable references
- honest evaluation status

over unsupported claims of accuracy or operational capability.

---

## Additional Documentation

For more detailed submission material, see:

```text
docs/
├── REQUIREMENTS.md
├── DATASET.md
├── ARCHITECTURE.md
├── VIDEO_SCRIPT.md
└── SUBMISSION_CHECKLIST.md
```

### Recommended reading order

For a reviewer:

```text
1. README.md
2. docs/ARCHITECTURE.md
3. docs/DATASET.md
4. docs/REQUIREMENTS.md
5. docs/VIDEO_SCRIPT.md
6. docs/SUBMISSION_CHECKLIST.md
```

---

## Quick Start

If you already have Node 24.x, pnpm, and Git installed:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd ASTRA-Vision-Vehicle-Recognition

pnpm install

cp .env.example .env

pnpm run typecheck
pnpm run build
```

Then start the API and frontend:

```bash
pnpm --filter @workspace/api-server run dev
```

In another terminal:

```bash
pnpm --filter @workspace/astra-vision run dev
```

Open the frontend URL printed by Vite and run an image analysis.

---

**ASTRA Vision — Vehicle Recognition & Evidence Review**

Built as a transparent, bounded computer-vision review prototype with catalog-aware evidence.
