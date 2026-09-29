import { Router, type IRouter } from "express";
import multer from "multer";
import {
  analyzeImage,
  catalogResponse,
  evaluationResponse,
  historyResponse,
  modelInfo,
} from "../services/vision";

const router: IRouter = Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
});

router.get("/model-info", async (_req, res, next) => {
  try {
    res.json(await modelInfo());
  } catch (error) {
    next(error);
  }
});

router.get("/evaluation", async (_req, res, next) => {
  try {
    res.json(await evaluationResponse());
  } catch (error) {
    next(error);
  }
});

router.get("/catalog", async (_req, res, next) => {
  try {
    res.json(await catalogResponse());
  } catch (error) {
    next(error);
  }
});

router.get("/history", async (_req, res, next) => {
  try {
    res.json(await historyResponse());
  } catch (error) {
    next(error);
  }
});

router.post("/analyze", upload.single("image"), async (req, res, next) => {
  try {
    if (!req.file) {
      res.status(400).json({ error: "Attach an image in the image field.", code: "IMAGE_REQUIRED" });
      return;
    }
    res.json(await analyzeImage(req.file.buffer, req.file.mimetype, req.file.originalname));
  } catch (error) {
    next(error);
  }
});

export default router;