import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, Link } from "react-router-dom";

import {
  Activity,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  CircleDot,
  FileImage,
  Network,
  ScanLine,
  ShieldCheck,
} from "lucide-react";

import {
  firstDefined,
  firstPath,
  getStoredAnalysisResult,
  imageSource,
  storeAnalysisResult,
} from "../../services/visionxApi";

/* =========================================================
   VISIONX — ANALYSIS WORKSPACE

   Visual system follows Home.jsx exactly:
   #F8FAFC / #0F172A / #334155 / #64748B
   #E2E8F0 / #0F766E / #059669
========================================================= */

const COLORS = {
  background: "#F8FAFC",
  heading: "#0F172A",
  secondary: "#334155",
  muted: "#64748B",
  border: "#E2E8F0",
  soft: "#F1F5F9",
  accent: "#0F766E",
  success: "#059669",
  dark: "#111816",
  dark2: "#1C2723",
};

const analysisStages = [
  {
    id: 1,
    title: "Quality gate",
    text: "Image adequacy",
    icon: ScanLine,
  },
  {
    id: 2,
    title: "Retinal structure",
    text: "Vessel & anatomy",
    icon: Activity,
  },
  {
    id: 3,
    title: "DR grading",
    text: "Severity classification",
    icon: CircleDot,
  },
  {
    id: 4,
    title: "Evidence",
    text: "Prepare explanation",
    icon: ShieldCheck,
  },
];

/* =========================================================
   TOP STEPS
========================================================= */

function StepIndicator() {
  const steps = [
    ["Image", true],
    ["Quality", true],
    ["Analysis", true],
    ["Explain", false],
    ["Review", false],
  ];

  return (
    <div className="hidden items-center xl:flex">
      {steps.map(([label, complete], index) => (
        <div key={label} className="flex items-center">
          <div className="flex items-center gap-2.5">
            <div
              className={`flex h-9 w-9 items-center justify-center rounded-xl border ${
                complete && label !== "Analysis"
                  ? "border-[#A7CBB9] bg-[#EAF5EE] text-[#3F7C5E]"
                  : label === "Analysis"
                    ? "border-[#111816] bg-[#111816] text-white"
                    : "border-[#E2E8F0] bg-white text-[#94A3B8]"
              }`}
            >
              {complete && label !== "Analysis" ? (
                <CheckCircle2 size={16} />
              ) : (
                <span className="text-[10px] font-black">
                  {index + 1}
                </span>
              )}
            </div>

            <span
              className={`text-[11px] font-bold ${
                label === "Analysis"
                  ? "text-[#0F172A]"
                  : complete
                    ? "text-[#334155]"
                    : "text-[#94A3B8]"
              }`}
            >
              {label}
            </span>
          </div>

          {index < steps.length - 1 && (
            <div
              className={`mx-4 h-px w-8 ${
                index < 2
                  ? "bg-[#A9C7B7]"
                  : "bg-[#E2E8F0]"
              }`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

/* =========================================================
   STAGE CARD
========================================================= */

function StageCard({ stage, index, active }) {
  const Icon = stage.icon;

  return (
    <div
      className={`rounded-2xl border p-5 transition ${
        active
          ? "border-[#A9D5C4] bg-[#F0FDFA]"
          : "border-[#E2E8F0] bg-white"
      }`}
    >
      <div className="flex items-center justify-between">
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-xl ${
            active
              ? "bg-[#0F766E] text-white"
              : "bg-[#F1F5F9] text-[#64748B]"
          }`}
        >
          <span className="text-sm font-black">●</span>
        </div>

        <span className="text-[10px] font-bold tracking-[0.18em] text-[#94A3B8]">
          0{index + 1}
        </span>
      </div>

      <div className="mt-6 text-[13px] font-bold text-[#0F172A]">
        {stage.title}
      </div>

      <div className="mt-1 text-[10px] font-medium text-[#64748B]">
        {stage.text}
      </div>

      <div className="mt-4 flex items-center gap-2">
        {active ? (
          <>
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#059669]" />
            <span className="text-[9px] font-bold uppercase tracking-[0.12em] text-[#0F766E]">
              Returned
            </span>
          </>
        ) : (
          <>
            <span className="h-1.5 w-1.5 rounded-full bg-[#CBD5E1]" />
            <span className="text-[9px] font-bold uppercase tracking-[0.12em] text-[#94A3B8]">
              Queued
            </span>
          </>
        )}
      </div>
    </div>
  );
}

/* =========================================================
   IMAGE VIEWER
========================================================= */

function ImageViewer({ previewUrl, backendImage, fileName }) {
  const source = backendImage || previewUrl;

  return (
    <div className="relative min-h-[560px] overflow-hidden rounded-[28px] bg-[#020617]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.035),transparent_58%)]" />

      {source ? (
        <img src={source} alt="Retinal fundus analysis" className="absolute inset-0 h-full w-full object-contain" />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <FileImage size={38} className="mx-auto text-white/25" />
            <div className="mt-4 text-[12px] font-bold text-white/60">Original image not available</div>
          </div>
        </div>
      )}

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_42%,rgba(0,0,0,0.40)_100%)]" />

      <div className="absolute left-4 top-4 rounded-2xl border border-white/15 bg-black/55 px-4 py-3 backdrop-blur-md">
        <div className="text-[9px] font-bold uppercase tracking-[0.2em] text-white/45">Analysis viewer</div>
        <div className="mt-1 max-w-[280px] truncate text-[12px] font-bold text-white">{fileName || "Retinal image"}</div>
      </div>

      <div className="absolute right-4 top-4 rounded-full border border-white/15 bg-black/50 px-4 py-2 text-[9px] font-bold uppercase tracking-[0.14em] text-white/60">API response</div>

      <div className="absolute bottom-4 left-4 rounded-lg border border-white/15 bg-black/45 px-3 py-2 text-[9px] font-bold uppercase tracking-[0.13em] text-white/50 backdrop-blur">Fundus image</div>
    </div>
  );
}

/* =========================================================
   RESULT PLACEHOLDER
========================================================= */

function ResultPanel({ result }) {
  const grade = firstDefined(firstPath(result, [
    "analysis.grade", "analysis.dr_grade", "analysis.drGrade",
    "grading.grade", "grading.dr_grade", "grading.drGrade",
    "dr_grade", "drGrade", "grade",
  ]));

  const confidence = firstDefined(firstPath(result, [
    "analysis.confidence", "grading.confidence", "confidence",
    "calibrated_confidence", "calibratedConfidence",
  ]));

  const quality = firstPath(result, [
    "quality", "quality_assessment", "qualityAssessment",
    "image_quality", "imageQuality",
  ]);

  return (
    <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
      <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#64748B]">Model output</div>
      <div className="mt-3 text-2xl font-black tracking-tight text-[#0F172A]">{grade !== undefined ? String(grade) : "No grade returned"}</div>
      <p className="mt-2 text-[12px] leading-5 text-[#64748B]">Values below are read directly from the connected VisionX inference response.</p>

      <div className="mt-6 grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] p-4">
          <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">ICDR grade</div>
          <div className="mt-2 text-2xl font-black text-[#0F172A]">{grade !== undefined ? String(grade) : "—"}</div>
        </div>
        <div className="rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] p-4">
          <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">Confidence</div>
          <div className="mt-2 text-2xl font-black text-[#0F172A]">{confidence !== undefined ? String(confidence) : "—"}</div>
        </div>
      </div>

      <div className="mt-3 rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] p-4">
        <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">Quality response</div>
        <div className="mt-2 text-[12px] font-bold text-[#334155]">{quality !== undefined ? "Returned" : "Not returned"}</div>
      </div>

      <div className="mt-4 flex items-start gap-3 rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] p-4">
        <ShieldCheck size={16} className="mt-0.5 shrink-0 text-[#0F766E]" />
        <div className="text-[10px] leading-5 text-[#64748B]">No prediction is generated by the frontend.</div>
      </div>
    </div>
  );
}

/* =========================================================
   MAIN
========================================================= */

export default function Analysis() {
  const navigate = useNavigate();
  const location = useLocation();

  const resultFromState = location.state?.analysisResult || null;
  const file = location.state?.file || null;
  const fileName = location.state?.fileName || file?.name || "";

  const [result, setResult] = useState(resultFromState || getStoredAnalysisResult());
  const [previewUrl, setPreviewUrl] = useState("");

  useEffect(() => {
    if (resultFromState) {
      storeAnalysisResult(resultFromState);
      setResult(resultFromState);
    }
  }, [resultFromState]);

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const quality = firstPath(result, ["quality", "quality_assessment", "qualityAssessment", "image_quality", "imageQuality"]);
  const structure = firstPath(result, ["structure", "retinal_structure", "retinalStructure", "segmentation"]);
  const grade = firstDefined(firstPath(result, ["analysis.grade", "analysis.dr_grade", "analysis.drGrade", "grading.grade", "grading.dr_grade", "grading.drGrade", "dr_grade", "drGrade", "grade"]));
  const explainability = firstPath(result, ["explainability", "xai", "evidence"]);
  const caseId = firstDefined(firstPath(result, ["case_id", "caseId", "id", "case.id", "metadata.case_id", "metadata.caseId"]));
  const backendImage = imageSource(firstPath(result, ["original_image_url", "originalImageUrl", "images.original", "image.original"]));

  const stages = [
    ["01", "Quality gate", quality !== undefined],
    ["02", "Retinal structure", structure !== undefined],
    ["03", "DR grading", grade !== undefined],
    ["04", "Evidence", explainability !== undefined],
  ];

  return (
    <main className="min-h-screen overflow-hidden bg-[#F8FAFC] text-[#0F172A]">
      <div className="pointer-events-none fixed inset-0 opacity-[0.38]" style={{ backgroundImage: "linear-gradient(#E2E8F0 1px, transparent 1px), linear-gradient(90deg, #E2E8F0 1px, transparent 1px)", backgroundSize: "72px 72px" }} />
      <div className="relative z-10">
        <header className="border-b border-[#E2E8F0] bg-white/95 backdrop-blur-md">
          <div className="mx-auto flex max-w-[1460px] items-center justify-between px-6 py-5 md:px-10 lg:px-12">
            <Link to="/" className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#0F766E] shadow-sm"><ScanLine size={20} /></div>
              <div><div className="text-[17px] font-black tracking-[0.29em] text-[#0F172A]">VISIONX</div><div className="mt-0.5 text-[9px] font-bold uppercase tracking-[0.22em] text-[#64748B]">Analysis Workspace</div></div>
            </Link>
            <StepIndicator />
            <button type="button" onClick={() => navigate("/screening")} className="inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-2.5 text-[10px] font-bold text-[#64748B] transition hover:border-[#CBD5E1] hover:text-[#334155]"><ArrowLeft size={13} />Back</button>
          </div>
        </header>

        <div className="mx-auto max-w-[1460px] px-6 py-10 md:px-10 lg:px-12">
          <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">Step 03 • Retinal analysis</div>
              <h1 className="mt-4 text-[42px] font-black leading-none tracking-tight text-[#0F172A] md:text-[52px]">Analyse the<span className="block text-[#0F766E]">retinal image.</span></h1>
              <p className="mt-6 max-w-[720px] text-[16px] font-medium leading-relaxed text-[#475569]">This workspace renders the actual response returned by the VisionX inference service.</p>
            </div>
            <div className="flex items-center gap-3 rounded-2xl border border-[#E2E8F0] bg-white px-5 py-4 shadow-sm">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]"><Network size={16} /></div>
              <div><div className="text-[9px] font-bold uppercase tracking-[0.16em] text-[#94A3B8]">Case</div><div className="mt-1 text-[12px] font-bold text-[#334155]">{caseId || "—"}</div></div>
            </div>
          </div>

          {!result ? (
            <section className="mt-8 rounded-[30px] border border-[#E2E8F0] bg-white p-16 text-center shadow-sm">
              <FileImage size={34} className="mx-auto text-[#CBD5E1]" />
              <h2 className="mt-5 text-xl font-black text-[#0F172A]">No analysis response</h2>
              <p className="mx-auto mt-2 max-w-[560px] text-[12px] leading-6 text-[#64748B]">Submit an image through Screening and return with a real inference response.</p>
              <button type="button" onClick={() => navigate("/screening")} className="mt-6 inline-flex items-center gap-2 rounded-full bg-[#111816] px-5 py-3 text-[11px] font-bold text-white">Go to screening<ArrowRight size={14} /></button>
            </section>
          ) : (
            <>
              <section className="mt-8 rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm">
                <div className="flex flex-col justify-between gap-3 md:flex-row md:items-center">
                  <div><div className="text-[10px] font-bold uppercase tracking-[0.20em] text-[#64748B]">Inference response</div><div className="mt-2 flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-[#059669]" /><span className="text-[14px] font-bold text-[#0F172A]">Response received</span></div></div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#64748B]">API-backed result</div>
                </div>
              </section>

              <div className="mt-6 grid gap-6 xl:grid-cols-[1.33fr_0.67fr]">
                <section className="rounded-[30px] border border-[#E2E8F0] bg-white p-4 shadow-sm md:p-5">
                  <div className="flex items-center justify-between px-1 pb-5"><div><div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#64748B]">Retinal imaging</div><h2 className="mt-2 text-xl font-black text-[#0F172A]">Fundus viewer</h2></div><div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">Returned image</div></div>
                  <ImageViewer previewUrl={previewUrl} backendImage={backendImage} fileName={fileName} />
                </section>

                <aside className="space-y-3">
                  {stages.map(([number, title, returned]) => (
                    <StageCard key={number} stage={{ title, text: returned ? "Returned by service" : "Not returned" }} index={Number(number) - 1} active={returned} />
                  ))}
                  <ResultPanel result={result} />
                </aside>
              </div>

              <section className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm md:p-7">
                <div><div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#0F766E]">Response inspection</div><h2 className="mt-3 text-2xl font-black tracking-tight text-[#0F172A]">What VisionX returned.</h2><p className="mt-2 max-w-[700px] text-[12px] leading-5 text-[#64748B]">Missing fields remain unavailable until the backend supplies them.</p></div>
                <div className="mt-6 grid gap-3 md:grid-cols-3">
                  {[["Quality", quality !== undefined],["Analysis", grade !== undefined || structure !== undefined],["Explainability", explainability !== undefined]].map(([label, returned]) => (
                    <div key={label} className="rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] p-5"><div className="text-[10px] font-bold uppercase tracking-[0.16em] text-[#64748B]">{label}</div><div className="mt-3 flex items-center gap-2 text-[11px] font-bold text-[#334155]">{returned ? <CheckCircle2 size={14} className="text-[#059669]" /> : <CircleDot size={14} className="text-[#CBD5E1]" />}{returned ? "Returned" : "Not returned"}</div></div>
                  ))}
                </div>
              </section>

              <section className="mt-8 overflow-hidden rounded-[30px] bg-[#111816] p-6 shadow-xl md:p-7">
                <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-center">
                  <div><div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#80958B]">Next stage</div><h2 className="mt-3 text-2xl font-black tracking-tight text-white">Inspect the returned<span className="text-[#80958B]"> evidence.</span></h2><p className="mt-3 max-w-[650px] text-[12px] leading-6 text-[#A9BEB4]">Open the separate explainability workspace to inspect XAI assets supplied by the service.</p></div>
                  <button type="button" onClick={() => navigate("/screening/explainability", { state: { analysisResult: result, file, fileName } })} className="inline-flex shrink-0 items-center justify-center gap-3 rounded-full bg-white px-6 py-3.5 text-[12px] font-bold text-[#111816] transition hover:bg-[#F8FAFC]">Open Explainability<ArrowRight size={15} /></button>
                </div>
              </section>
            </>
          )}

          <div className="mt-10 flex flex-col justify-between gap-3 border-t border-[#E2E8F0] pt-6 text-[10px] font-medium text-[#64748B] md:flex-row"><div>VisionX Analysis Workspace • API response viewer</div><div className="flex items-center gap-2"><ShieldCheck size={13} />No client-side prediction</div></div>
        </div>
      </div>
    </main>
  );
}
