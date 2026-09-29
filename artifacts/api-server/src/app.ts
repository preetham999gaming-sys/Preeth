import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes";
import { logger } from "./lib/logger";

const app: Express = express();

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return {
          id: req.id,
          method: req.method,
          url: req.url?.split("?")[0],
        };
      },
      res(res) {
        return {
          statusCode: res.statusCode,
        };
      },
    },
  }),
);
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.use("/api", router);

app.use((error: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  logger.error({ err: error }, "Unhandled API error");
  const message = error instanceof Error ? error.message : "";
  const clientError = /empty|exceeds|unsupported|could not be decoded|attach an image/i.test(message);
  res.status(clientError ? 400 : 500).json({
    error: clientError ? message : "The analysis service encountered an unexpected error.",
    code: clientError ? "INVALID_IMAGE" : "INTERNAL_ERROR",
  });
});

export default app;
