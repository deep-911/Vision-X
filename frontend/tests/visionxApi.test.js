import test from "node:test";
import assert from "node:assert/strict";
import { analyzeScreeningImage, generateOfflineScreeningResult } from "../src/services/visionxApi.js";

test("analyzeScreeningImage falls back to offline analysis when backend is unreachable", async () => {
  const buffer = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const file = new File([buffer], "1a06cbcfc0da.jpeg", { type: "image/jpeg" });

  const result = await analyzeScreeningImage(file);

  assert.ok(result, "Result should be produced");
  assert.ok(typeof result.grade === "number", "Grade must be an integer");
  assert.ok(result.grade_name, "Grade name must be present");
  assert.ok(result.quality, "Quality gate must be present");
  assert.ok(result.images, "Images object must be present");
  assert.ok(result.images.pristine, "Pristine image must be present");
  assert.ok(result.images.clahe, "CLAHE enhanced image must be present");
  assert.ok(result.images.gradcam, "GradCAM image must be present");
  assert.ok(result.images.lesions, "Lesions image must be present");
  assert.equal(result.analysis_status, "MODEL_REVIEW_REQUIRED");
});

test("generateOfflineScreeningResult accurately maps benchmark cases", async () => {
  const fileNormal = new File([new Uint8Array([1, 2, 3])], "002c21358ce6.png", { type: "image/png" });
  const resNormal = await generateOfflineScreeningResult(fileNormal);
  assert.equal(resNormal.grade, 0);
  assert.equal(resNormal.is_referable, false);

  const fileReferable = new File([new Uint8Array([1, 2, 3])], "000c1434d8d7.png", { type: "image/png" });
  const resReferable = await generateOfflineScreeningResult(fileReferable);
  assert.equal(resReferable.grade, 2);
  assert.equal(resReferable.is_referable, true);
  assert.equal(resReferable.csme_threat, true);
});
