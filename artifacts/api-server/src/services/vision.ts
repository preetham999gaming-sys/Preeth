import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { performance } from "node:perf_hooks";

import { pipeline, RawImage } from "@huggingface/transformers";
import sharp from "sharp";

import {
  AnalyzeImageResponse,
  GetCatalogResponse,
  GetEvaluationResponse,
  GetHistoryResponse,
  GetModelInfoResponse,
} from "@workspace/api-zod";
import { logger } from "../lib/logger";

const DATA_ROOT = [
  path.resolve(process.cwd(), "artifacts/astra-vision/public/dataset"),
  path.resolve(process.cwd(), "../astra-vision/public/dataset"),
  path.resolve(process.cwd(), "../../artifacts/astra-vision/public/dataset"),
].find((candidate) => existsSync(candidate)) ?? path.resolve(process.cwd(), "artifacts/astra-vision/public/dataset");
const IMAGE_ROOT = path.join(DATA_ROOT, "images");
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024;
const REFERENCE_REPOSITORY = "https://github.com/AdvaithVellanki/Aircraft-Classification";
const MODEL_NAME = "Xenova/clip-vit-base-patch32";
const CLASS_NAMES = ["aircraft", "helicopter", "drone", "military-vehicle", "naval"] as const;
const CLASS_LABELS = ["aircraft", "helicopter", "drone", "military vehicle", "naval vessel"] as const;
const DISPLAY_LABELS = {
  aircraft: "Aircraft",
  helicopter: "Helicopter",
  drone: "Drone",
  "military-vehicle": "Military Vehicle",
  naval: "Naval Vessel",
} as const;

type Category = (typeof CLASS_NAMES)[number];

type CatalogRow = {
  fileName: string;
  category: Category;
  sourceTitle: string;
  artist: string;
  license: string;
  sourceUrl: string;
};

type Reference = CatalogRow & {
  id: string;
  exactName: string;
  sha256?: string;
  signature?: Float32Array;
};

type Classification = { label: string; score: number };

const history: Array<{
  id: string;
  analyzed_at: Date;
  file_name: string;
  category: string;
  confidence: number;
  evidence: string;
}> = [];

let catalogPromise: Promise<Reference[]> | undefined;
let classifierPromise: Promise<any> | undefined;

function parseCsvLine(line: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    const next = line[index + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      cells.push(cell);
      cell = "";
    } else {
      cell += char;
    }
  }
  cells.push(cell);
  return cells;
}

function parseCsv(text: string): Record<string, string>[] {
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function cleanTitle(value: string): string {
  return value
    .replace(/^File:/i, "")
    .replace(/\.(jpg|jpeg|png|webp)$/i, "")
    .replace(/[-\s](jpg|jpeg|png|webp)$/i, "")
    .replace(/\s*\([^)]*\)\s*$/g, "")
    .replace(/_+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function exactNameFor(row: CatalogRow): string {
  const title = cleanTitle(row.sourceTitle);
  const fileBase = cleanTitle(path.basename(row.fileName));
  const knownNames = [
    "F-22 Raptor",
    "F-22",
    "F/A-18 Super Hornet",
    "MiG-29K",
    "MiG-21",
    "Su-47",
    "HAL TEDBF",
    "J-50",
    "SGR-01",
    "G4M-J1N",
    "Vindkast Mk.2E2",
    "Bristol Beaufighter",
    "Douglas A-20 Havoc",
    "DH113 Vampire",
    "Dornier Do 217N",
    "F2H-2N",
    "F3D-2 Skynight",
    "F7F-3N",
    "Avro 504",
    "CH-47 Chinook",
    "T-84",
    "Renault FT-17",
    "PzKpfw 35(t)",
    "T-44",
    "USS Batfish",
    "HMCS St. Laurent",
    "Narwhal",
  ];
  const candidate = `${title} ${fileBase}`;
  const compact = (value: string) => value.toLowerCase().replace(/[^a-z0-9]+/g, "");
  const compactCandidate = compact(candidate);
  const known = knownNames.find((name) => compactCandidate.includes(compact(name)));
  if (known) return known.replace("FA18", "F/A-18");
  if (row.category === "aircraft" && fileBase) return fileBase.replace(/^\d+[-\s]+/, "");
  return title || `${DISPLAY_LABELS[row.category]} reference`;
}

async function signatureFor(buffer: Buffer): Promise<Float32Array> {
  const raw = await sharp(buffer)
    .rotate()
    .resize(24, 24, { fit: "fill" })
    .grayscale()
    .raw()
    .toBuffer();
  const values = new Float32Array(raw.length);
  let mean = 0;
  for (let index = 0; index < raw.length; index += 1) mean += raw[index];
  mean /= raw.length || 1;
  let magnitude = 0;
  for (let index = 0; index < raw.length; index += 1) {
    values[index] = (raw[index] - mean) / 255;
    magnitude += values[index] ** 2;
  }
  const normalizer = Math.sqrt(magnitude) || 1;
  for (let index = 0; index < values.length; index += 1) values[index] /= normalizer;
  return values;
}

function cosineSimilarity(left: Float32Array, right: Float32Array): number {
  let score = 0;
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    score += left[index] * right[index];
  }
  return Math.max(0, Math.min(1, (score + 1) / 2));
}

async function loadCatalog(): Promise<Reference[]> {
  const [labelsText, creditsText] = await Promise.all([
    readFile(path.join(DATA_ROOT, "labels.csv"), "utf8"),
    readFile(path.join(DATA_ROOT, "credits.csv"), "utf8"),
  ]);
  const labels = parseCsv(labelsText);
  const credits = new Map(parseCsv(creditsText).map((row) => [row.file_name, row]));
  const references: Reference[] = [];

  for (const label of labels) {
    const category = label.category as Category;
    if (!CLASS_NAMES.includes(category)) continue;
    const credit = credits.get(label.file_name) ?? {};
    const row: CatalogRow = {
      fileName: label.file_name,
      category,
      sourceTitle: credit.source_title ?? label.file_name,
      artist: credit.artist ?? "Not listed",
      license: credit.license ?? "Not listed",
      sourceUrl: credit.commons_page ?? "",
    };
    const absolutePath = path.join(DATA_ROOT, row.fileName);
    try {
      const buffer = await readFile(absolutePath);
      references.push({
        ...row,
        id: createHash("sha1").update(row.fileName).digest("hex").slice(0, 12),
        exactName: exactNameFor(row),
        sha256: createHash("sha256").update(buffer).digest("hex"),
        signature: await signatureFor(buffer),
      });
    } catch (error) {
      logger.warn({ err: error, file: row.fileName }, "Skipping unreadable catalog image");
    }
  }
  logger.info({ references: references.length }, "Reference catalog loaded");
  return references;
}

async function getCatalog(): Promise<Reference[]> {
  if (!catalogPromise) {
    catalogPromise = loadCatalog().catch((error) => {
      catalogPromise = undefined;
      throw error;
    });
  }
  return catalogPromise;
}

async function getClassifier(): Promise<any> {
  if (!classifierPromise) {
    logger.info({ model: MODEL_NAME }, "Loading zero-shot image classifier");
    classifierPromise = pipeline("zero-shot-image-classification", MODEL_NAME, {
      device: "cpu",
    });
  }
  try {
    return await classifierPromise;
  } catch (error) {
    classifierPromise = undefined;
    logger.error({ err: error, model: MODEL_NAME }, "Zero-shot classifier failed to load");
    throw error;
  }
}

function categoryFromLabel(label: string): Category {
  const normalized = label.toLowerCase();
  if (normalized.includes("helicopter")) return "helicopter";
  if (normalized.includes("drone") || normalized.includes("uav")) return "drone";
  if (normalized.includes("ship") || normalized.includes("naval") || normalized.includes("vessel")) return "naval";
  if (normalized.includes("vehicle") || normalized.includes("tank") || normalized.includes("armour")) return "military-vehicle";
  return "aircraft";
}

function roundConfidence(value: number): number {
  return Number(Math.max(0, Math.min(1, value)).toFixed(4));
}

function confidenceStatus(confidence: number, exactMatch: boolean): "high" | "medium" | "low" | "unsupported" {
  if (exactMatch) return "high";
  if (confidence >= 0.72) return "high";
  if (confidence >= 0.5) return "medium";
  if (confidence >= 0.3) return "low";
  return "unsupported";
}

function toApiReference(reference: Reference, similarity: number, evidenceLevel: "exact-file-match" | "visual-neighbor" | "catalog-reference") {
  return {
    exact_name: reference.exactName,
    category: reference.category,
    reference_file: reference.fileName,
    reference_title: reference.sourceTitle,
    source_url: reference.sourceUrl,
    license: reference.license,
    similarity: roundConfidence(similarity),
    evidence_level: evidenceLevel,
  };
}

export async function modelInfo() {
  const catalog = await getCatalog();
  return GetModelInfoResponse.parse({
    name: "ASTRA catalog-aware vision",
    classifier: MODEL_NAME,
    task: "five-class zero-shot classification with evidence-backed catalog matching",
    reference_repository: REFERENCE_REPOSITORY,
    supported_classes: CLASS_NAMES,
    dataset_images: catalog.length,
    catalog_models: new Set(catalog.map((entry) => entry.exactName)).size,
    input_formats: ["jpeg", "png", "webp"],
    max_file_size_mb: MAX_FILE_SIZE_BYTES / (1024 * 1024),
    limitations: [
      "The starter labels are image-level categories, not object bounding boxes, so localization is not claimed.",
      "Exact vehicle names are returned only when supported by a catalog file match or a clearly labeled visual neighbor.",
      "The five-class dataset is small and mixed-quality; evaluation metrics remain Not evaluated until a held-out model run is recorded.",
      "Model scores are confidence signals, not operational conclusions.",
    ],
  });
}

export async function isModelLoaded(): Promise<boolean> {
  try {
    await getCatalog();
    return true;
  } catch {
    return false;
  }
}

export async function catalogResponse() {
  const catalog = await getCatalog();
  const classCounts = Object.fromEntries(CLASS_NAMES.map((name) => [name, catalog.filter((entry) => entry.category === name).length]));
  return GetCatalogResponse.parse({
    entries: catalog.map((entry) => ({
      id: entry.id,
      exact_name: entry.exactName,
      category: entry.category,
      file_name: entry.fileName,
      image_url: `/dataset/${entry.fileName}`,
      source_title: entry.sourceTitle,
      artist: entry.artist,
      license: entry.license,
      source_url: entry.sourceUrl,
    })),
    class_counts: classCounts,
  });
}

export async function evaluationResponse() {
  const catalog = await getCatalog();
  const classDistribution = Object.fromEntries(CLASS_NAMES.map((name) => [name, catalog.filter((entry) => entry.category === name).length]));
  const hashes = catalog.map((entry) => entry.sha256).filter(Boolean);
  const duplicateImages = hashes.length - new Set(hashes).size;
  return GetEvaluationResponse.parse({
    evaluated: false,
    dataset_report: {
      total_images: catalog.length,
      classes: CLASS_NAMES.length,
      split: "Planned stratified split: 80% train / 10% validation / 10% test",
      train_images: Math.round(catalog.length * 0.8),
      validation_images: Math.round(catalog.length * 0.1),
      test_images: catalog.length - Math.round(catalog.length * 0.8) - Math.round(catalog.length * 0.1),
      invalid_images: 0,
      duplicate_images: duplicateImages,
      class_distribution: classDistribution,
      source: "ASTRA Challenge Starter / Wikimedia Commons credits.csv",
    },
    metrics: {
      accuracy: null,
      top3_accuracy: null,
      precision: null,
      recall: null,
      f1: null,
    },
    per_class: CLASS_NAMES.map((label) => ({
      label,
      image_count: classDistribution[label],
      accuracy: null,
    })),
    note: "Not evaluated: this build has the labeled starter set and a live zero-shot baseline, but no held-out fine-tuned checkpoint. No accuracy is fabricated.",
  });
}

export async function historyResponse() {
  return GetHistoryResponse.parse({ items: history.slice(0, 30) });
}

export async function analyzeImage(buffer: Buffer, mimeType: string, fileName: string) {
  if (!buffer.length) throw new Error("The uploaded image is empty.");
  if (buffer.byteLength > MAX_FILE_SIZE_BYTES) throw new Error("The uploaded image exceeds the 10 MB limit.");
  if (!["image/jpeg", "image/png", "image/webp"].includes(mimeType)) throw new Error("Unsupported image format. Use JPG, PNG, or WEBP.");

  const metadata = await sharp(buffer).metadata();
  if (!metadata.width || !metadata.height || !metadata.format) throw new Error("The image could not be decoded.");
  const normalized = await sharp(buffer).rotate().png().toBuffer();
  const startedAt = performance.now();
  const catalog = await getCatalog();
  const sha256 = createHash("sha256").update(buffer).digest("hex");
  const exact = catalog.find((entry) => entry.sha256 === sha256);
  const uploadedSignature = await signatureFor(normalized);
  const neighbors = catalog
    .filter((entry) => entry.signature)
    .map((entry) => ({ entry, similarity: cosineSimilarity(uploadedSignature, entry.signature!) }))
    .sort((left, right) => right.similarity - left.similarity);
  const bestNeighbor = neighbors[0];

  let topPredictions: Array<{ label: string; confidence: number; rank: number }>;
  let category: Category;
  let confidence: number;
  let referenceMatch: ReturnType<typeof toApiReference> | null = null;
  let mode: string;
  let explanation: string;

  if (exact) {
    category = exact.category;
    confidence = 1;
    topPredictions = [category, ...CLASS_NAMES.filter((item) => item !== category)]
      .slice(0, 3)
      .map((label, index) => ({ label: DISPLAY_LABELS[label], confidence: roundConfidence(index === 0 ? 1 : 0), rank: index + 1 }));
    referenceMatch = toApiReference(exact, 1, "exact-file-match");
    mode = "exact catalog evidence";
    explanation = `This file matches the ASTRA reference catalog exactly. The vehicle name is reported from the credited source entry: ${exact.exactName}.`;
  } else {
    try {
      const classifier = await getClassifier();
      const image = await RawImage.read(new Blob([new Uint8Array(normalized)], { type: "image/png" }));
      const raw = (await classifier(image, [...CLASS_LABELS])) as Classification[];
      topPredictions = raw.map((item, index) => {
        const itemCategory = categoryFromLabel(item.label);
        return { label: DISPLAY_LABELS[itemCategory], confidence: roundConfidence(item.score), rank: index + 1 };
      });
      const deduped = Array.from(new Map(topPredictions.map((item) => [item.label, item])).values()).slice(0, 3);
      topPredictions = deduped.map((item, index) => ({ ...item, rank: index + 1 }));
      const best = topPredictions[0] ?? { label: "Unsupported / unclear", confidence: 0, rank: 1 };
      category = categoryFromLabel(best.label);
      confidence = best.confidence;
      mode = "zero-shot classifier";
      if (bestNeighbor && bestNeighbor.similarity >= 0.82) {
        referenceMatch = toApiReference(bestNeighbor.entry, bestNeighbor.similarity, "visual-neighbor");
      }
      explanation = referenceMatch
        ? `The live classifier selected ${DISPLAY_LABELS[category]}. A visually similar credited reference is shown separately; it is not proof that the uploaded vehicle is the same model.`
        : `The live classifier selected ${DISPLAY_LABELS[category]}. No sufficiently strong catalog neighbor supports an exact vehicle name, so the result stays at the class level.`;
    } catch (error) {
      logger.warn({ err: error }, "Classifier unavailable; using reference-neighbor fallback");
      category = bestNeighbor?.entry.category ?? "aircraft";
      confidence = bestNeighbor ? Math.max(0.05, bestNeighbor.similarity * 0.65) : 0.05;
      topPredictions = [category, ...CLASS_NAMES.filter((item) => item !== category)]
        .slice(0, 3)
        .map((label, index) => ({ label: DISPLAY_LABELS[label], confidence: roundConfidence(index === 0 ? confidence : 0), rank: index + 1 }));
      mode = "reference-neighbor fallback";
      if (bestNeighbor && bestNeighbor.similarity >= 0.68) {
        referenceMatch = toApiReference(bestNeighbor.entry, bestNeighbor.similarity, "visual-neighbor");
      }
      explanation = "The zero-shot classifier is unavailable in this runtime. The class signal comes from visual similarity to the credited starter catalog and is marked low-confidence.";
    }
  }

  const elapsed = Math.max(1, Math.round(performance.now() - startedAt));
  const result = AnalyzeImageResponse.parse({
    prediction: topPredictions[0],
    top_predictions: topPredictions,
    reference_match: referenceMatch,
    detections: [{
      class_name: DISPLAY_LABELS[category],
      confidence: roundConfidence(confidence),
      box: null,
    }],
    image: {
      width: metadata.width,
      height: metadata.height,
      format: metadata.format.toUpperCase(),
      size_bytes: buffer.byteLength,
    },
    preprocessing: {
      applied: true,
      summary: "Decoded, EXIF-rotated, converted to RGB PNG, resized for visual-neighbor matching, and sent to the classifier when available.",
    },
    model: {
      detector: "Not enabled — no bounding-box labels in the starter dataset",
      classifier: MODEL_NAME,
      mode,
      device: "cpu",
    },
    confidence_status: confidenceStatus(confidence, Boolean(exact)),
    explanation,
    inference_time_ms: elapsed,
    analyzed_at: new Date(),
  });

  history.unshift({
    id: createHash("sha1").update(`${fileName}:${result.analyzed_at.toISOString()}`).digest("hex").slice(0, 12),
    analyzed_at: result.analyzed_at,
    file_name: fileName,
    category: result.prediction.label,
    confidence: result.prediction.confidence,
    evidence: referenceMatch?.evidence_level ?? mode,
  });
  return result;
}