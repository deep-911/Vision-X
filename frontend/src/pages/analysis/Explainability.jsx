import { useEffect, useState } from "react";
import { useLocation, useNavigate, Link } from "react-router-dom";

import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  CircleDot,
  Eye,
  FileCheck2,
  Info,
  Network,
  ScanLine,
  ShieldCheck,
  Target,
} from "lucide-react";

import {
  firstDefined,
  firstPath,
  getStoredAnalysisResult,
  imageSource,
  storeAnalysisResult,
} from "../../services/visionxApi";

/* =========================================================
   VISIONX — EXPLAINABILITY WORKSPACE
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
  evidence: "#D7A553",
  risk: "#B76967",
};

/* =========================================================
   WORKFLOW HEADER
========================================================= */

function StepIndicator() {
  const steps = [
    ["Image", true],
    ["Quality", true],
    ["Analysis", true],
    ["Explain", true],
    ["Review", false],
  ];

  return (
    <div className="hidden items-center xl:flex">
      {steps.map(([label, complete], index) => (
        <div key={label} className="flex items-center">
          <div className="flex items-center gap-2.5">
            <div
              className={`flex h-9 w-9 items-center justify-center rounded-xl border ${
                label === "Explain"
                  ? "border-[#111816] bg-[#111816] text-white"
                  : complete
                    ? "border-[#A7CBB9] bg-[#EAF5EE] text-[#3F7C5E]"
                    : "border-[#E2E8F0] bg-white text-[#94A3B8]"
              }`}
            >
              {complete && label !== "Explain" ? (
                <CheckCircle2 size={16} />
              ) : (
                <span className="text-[10px] font-black">
                  {index + 1}
                </span>
              )}
            </div>

            <span
              className={`text-[11px] font-bold ${
                label === "Explain"
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
                index < 3
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
   EVIDENCE VIEWER
========================================================= */

function EvidenceViewer({ source, title, emptyText }) {
  return (
    <div className="relative min-h-[560px] overflow-hidden rounded-[28px] bg-[#020617]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.025),transparent_58%)]" />
      {source ? (
        <img src={source} alt={title} className="absolute inset-0 h-full w-full object-contain" />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center p-8 text-center">
          <div><Eye size={42} className="mx-auto text-white/20" /><div className="mt-4 text-[13px] font-bold text-white/60">{emptyText}</div><div className="mx-auto mt-2 max-w-[360px] text-[10px] leading-5 text-white/35">The inference response did not include an image asset for this view.</div></div>
        </div>
      )}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_42%,rgba(0,0,0,0.40)_100%)]" />
      <div className="absolute left-4 top-4 rounded-2xl border border-white/15 bg-black/55 px-4 py-3 backdrop-blur-md"><div className="text-[9px] font-bold uppercase tracking-[0.20em] text-white/45">Explainability view</div><div className="mt-1 text-[12px] font-bold text-white">{title}</div></div>
      <div className="absolute bottom-4 left-4 rounded-xl border border-white/15 bg-black/55 px-3 py-2 text-[9px] font-bold uppercase tracking-[0.15em] text-white/55 backdrop-blur-md">Backend asset</div>
    </div>
  );
}

/* =========================================================
   XAI RESULT CARD
========================================================= */

function EvidenceCard({
  icon: Icon,
  title,
  value,
  description,
  accent = false,
}) {
  return (
    <div
      className={`rounded-2xl border p-5 ${
        accent
          ? "border-[#D8E7DE] bg-[#F8FBF9]"
          : "border-[#E2E8F0] bg-white"
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
          <Icon size={17} />
        </div>

        {accent && (
          <CheckCircle2
            size={16}
            className="text-[#059669]"
          />
        )}
      </div>

      <div className="mt-5 text-[10px] font-bold uppercase tracking-[0.16em] text-[#64748B]">
        {title}
      </div>

      <div className="mt-2 text-[17px] font-black text-[#0F172A]">
        {value}
      </div>

      <p className="mt-2 text-[11px] leading-5 text-[#64748B]">
        {description}
      </p>
    </div>
  );
}

/* =========================================================
   MAIN
========================================================= */

export default function Explainability() {
  const navigate = useNavigate();
  const location = useLocation();

  const resultFromState = location.state?.analysisResult || null;
  const file = location.state?.file || null;
  const fileName = location.state?.fileName || file?.name || "";

  const [result, setResult] = useState(resultFromState || getStoredAnalysisResult());
  const [mode, setMode] = useState("original");
  const [previewUrl, setPreviewUrl] = useState("");

  useEffect(() => {
    if (resultFromState) {
      storeAnalysisResult(resultFromState);
      setResult(resultFromState);
    }
  }, [resultFromState]);

  useEffect(() => {
    if (!file) { setPreviewUrl(""); return undefined; }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const originalSource = imageSource(firstPath(result, ["original_image_url", "originalImageUrl", "images.original", "image.original"])) || previewUrl;
  const gradCamSource = imageSource(firstPath(result, ["explainability.gradcam_url", "explainability.gradcam", "xai.gradcam_url", "xai.gradcam", "gradcam_url", "gradcam", "images.gradcam", "images.grad_cam", "evidence.gradcam_url"]));
  const lesionSource = imageSource(firstPath(result, ["explainability.lesion_evidence_url", "explainability.lesionEvidenceUrl", "explainability.lesion_overlay_url", "explainability.lesionOverlayUrl", "xai.lesion_evidence_url", "xai.lesionEvidenceUrl", "lesion_evidence_url", "lesionEvidenceUrl", "lesion_overlay_url", "lesionOverlayUrl", "images.lesion_evidence", "images.lesions", "evidence.lesion_image_url"]));
  const lesions = firstDefined(firstPath(result, ["explainability.lesions", "xai.lesions", "lesions", "lesion_evidence", "lesionEvidence"]));
  const caseId = firstDefined(firstPath(result, ["case_id", "caseId", "id", "case.id", "metadata.case_id", "metadata.caseId"]));
  const grade = firstDefined(firstPath(result, ["analysis.grade", "analysis.dr_grade", "analysis.drGrade", "grading.grade", "grading.dr_grade", "grading.drGrade", "dr_grade", "drGrade", "grade"]));
  const confidence = firstDefined(firstPath(result, ["analysis.confidence", "grading.confidence", "confidence", "calibrated_confidence", "calibratedConfidence"]));

  const source = mode === "gradcam" ? gradCamSource : mode === "lesions" ? lesionSource : originalSource;
  const title = mode === "gradcam" ? "Grad-CAM attention" : mode === "lesions" ? "Lesion evidence" : "Original image";
  const available = { original: Boolean(originalSource), gradcam: Boolean(gradCamSource), lesions: Boolean(lesionSource) };

  return (
    <main className="min-h-screen overflow-hidden bg-[#F8FAFC] text-[#0F172A]">
      <div className="pointer-events-none fixed inset-0 opacity-[0.38]" style={{ backgroundImage: "linear-gradient(#E2E8F0 1px, transparent 1px), linear-gradient(90deg, #E2E8F0 1px, transparent 1px)", backgroundSize: "72px 72px" }} />
      <div className="relative z-10">
        <header className="border-b border-[#E2E8F0] bg-white/95 backdrop-blur-md">
          <div className="mx-auto flex max-w-[1460px] items-center justify-between px-6 py-5 md:px-10 lg:px-12">
            <Link to="/" className="flex items-center gap-3"><div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#0F766E] shadow-sm"><Eye size={20} /></div><div><div className="text-[17px] font-black tracking-[0.29em] text-[#0F172A]">VISIONX</div><div className="mt-0.5 text-[9px] font-bold uppercase tracking-[0.22em] text-[#64748B]">Explainability Workspace</div></div></Link>
            <StepIndicator />
            <button type="button" onClick={() => navigate("/screening/analysis", { state: { analysisResult: result, file, fileName } })} className="inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-2.5 text-[10px] font-bold text-[#64748B] transition hover:border-[#CBD5E1] hover:text-[#334155]"><ArrowLeft size={13} />Back</button>
          </div>
        </header>

        <div className="mx-auto max-w-[1460px] px-6 py-10 md:px-10 lg:px-12">
          <div className="max-w-[850px]"><div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">Step 04 • Explainability</div><h1 className="mt-4 text-[42px] font-black leading-none tracking-tight text-[#0F172A] md:text-[54px]">Inspect the<span className="block text-[#0F766E]">returned evidence.</span></h1><p className="mt-6 max-w-[720px] text-[16px] font-medium leading-relaxed text-[#475569]">This workspace displays only the explainability assets and values contained in the actual analysis response.</p></div>

          {!result ? (
            <section className="mt-8 rounded-[30px] border border-[#E2E8F0] bg-white p-16 text-center shadow-sm"><Eye size={34} className="mx-auto text-[#CBD5E1]" /><h2 className="mt-5 text-xl font-black text-[#0F172A]">No explainability response</h2><p className="mx-auto mt-2 max-w-[560px] text-[12px] leading-6 text-[#64748B]">Run a real screening inference first. This page does not simulate Grad-CAM or lesion evidence.</p><button type="button" onClick={() => navigate("/screening")} className="mt-6 inline-flex items-center gap-2 rounded-full bg-[#111816] px-5 py-3 text-[11px] font-bold text-white">Go to screening<ArrowRight size={14} /></button></section>
          ) : (
            <>
              <section className="mt-9 overflow-hidden rounded-[32px] bg-[#111816] p-5 shadow-2xl md:p-6"><div className="rounded-[24px] border border-[#2B3B35] bg-[#16201D] p-5 md:p-6">
                <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center"><div><div className="text-[10px] font-bold uppercase tracking-[0.23em] text-[#80958B]">Explainability workspace</div><div className="mt-2 text-[15px] font-bold text-white">{caseId || fileName || "—"}</div></div><div className="flex items-center gap-2 rounded-full border border-[#2B3B35] bg-[#1C2723] px-4 py-2 text-[9px] font-bold uppercase tracking-[0.13em] text-[#A9BEB4]"><span className="h-1.5 w-1.5 rounded-full bg-[#80958B]" />Backend evidence response</div></div>

                <div className="mt-6 grid gap-5 lg:grid-cols-[1.22fr_0.78fr]">
                  <div><EvidenceViewer source={source} title={title} emptyText={mode === "gradcam" ? "Grad-CAM asset not returned" : mode === "lesions" ? "Lesion evidence asset not returned" : "Original image not returned"} />
                    <div className="mt-3 flex flex-wrap gap-2">
                      {[["original","Original"],["gradcam","Grad-CAM"],["lesions","Lesion evidence"]].map(([id,label]) => <button key={id} type="button" disabled={!available[id]} onClick={() => setMode(id)} className={`rounded-xl border px-4 py-2.5 text-[10px] font-bold transition ${mode === id ? "border-white/20 bg-white text-[#111816]" : "border-[#2B3B35] bg-[#1C2723] text-[#A9BEB4] hover:bg-[#25312C]"} disabled:cursor-not-allowed disabled:opacity-35`}>{label}</button>)}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723]/55 p-5"><div className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#80958B]">Screening output</div><div className="mt-3 text-[32px] font-black text-white">{grade !== undefined ? String(grade) : "—"}</div><div className="mt-1 text-[11px] font-bold text-[#A9BEB4]">Returned grade</div></div>
                    <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723]/55 p-5"><div className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#80958B]">Evidence state</div><div className="mt-4 space-y-4">{[["Grad-CAM",Boolean(gradCamSource)],["Lesion evidence",Boolean(lesionSource || lesions)],["Confidence",confidence !== undefined]].map(([label,present]) => <div key={label} className="flex items-center justify-between"><span className="text-[11px] font-bold text-[#A9BEB4]">{label}</span><span className={`text-[10px] font-bold uppercase tracking-[0.12em] ${present ? "text-[#34D399]" : "text-[#80958B]"}`}>{present ? "Returned" : "Not returned"}</span></div>)}</div></div>
                  </div>
                </div>
              </div></section>

              <section className="mt-7 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                <EvidenceCard icon={Eye} title="Visual explanation" value={gradCamSource ? "Returned" : "Not returned"} description="Grad-CAM visualization supplied by the analysis response." accent={Boolean(gradCamSource)} />
                <EvidenceCard icon={Target} title="Lesion evidence" value={lesionSource || lesions ? "Returned" : "Not returned"} description="Lesion-level evidence supplied by the analysis response." accent={Boolean(lesionSource || lesions)} />
                <EvidenceCard icon={CircleDot} title="Confidence" value={confidence !== undefined ? String(confidence) : "Not returned"} description="Displayed only when provided by the inference response." accent={confidence !== undefined} />
                <EvidenceCard icon={FileCheck2} title="Case ID" value={caseId || "Not returned"} description="Backend-assigned identifier for downstream review." accent={Boolean(caseId)} />
              </section>

              <section className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm"><div><div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#0F766E]">Lesion details</div><h2 className="mt-3 text-2xl font-black tracking-tight text-[#0F172A]">Returned lesion evidence.</h2><p className="mt-2 max-w-[700px] text-[12px] leading-5 text-[#64748B]">Only lesion records present in the backend response are shown.</p></div>
                {!Array.isArray(lesions) ? <div className="mt-6 rounded-2xl bg-[#F8FAFC] p-5 text-[11px] text-[#64748B]">No lesion list was returned by the inference service.</div> : lesions.length === 0 ? <div className="mt-6 rounded-2xl bg-[#F8FAFC] p-5 text-[11px] text-[#64748B]">The inference service returned an empty lesion list.</div> : <div className="mt-6 space-y-2">{lesions.map((lesion,index) => { const label=lesion?.label ?? lesion?.type ?? lesion?.class ?? "—"; const c=lesion?.confidence ?? lesion?.score; const l=lesion?.location ?? lesion?.region; return <div key={`${label}-${index}`} className="rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] p-4"><div className="flex items-center justify-between gap-4"><div className="text-[11px] font-bold text-[#334155]">{label}</div><div className="text-[10px] font-bold text-[#64748B]">{c !== undefined ? String(c) : "—"}</div></div><div className="mt-1 text-[10px] text-[#64748B]">{l ? String(l) : "Location not returned"}</div></div>})}</div>}
              </section>

              <section className="mt-8 overflow-hidden rounded-[30px] bg-[#111816] px-6 py-9 shadow-xl md:px-8"><div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-center"><div><div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#80958B]">Next workflow stage</div><h2 className="mt-3 text-2xl font-black text-white">Send the case to<span className="text-[#80958B]"> clinical review.</span></h2><p className="mt-3 max-w-[620px] text-[12px] leading-6 text-[#A9BEB4]">The clinical review workspace consumes this analysis response and backend-assigned case ID.</p></div><button type="button" disabled={!caseId} onClick={() => navigate(`/doctor/review/${encodeURIComponent(String(caseId))}`, { state: { analysisResult: result, file, fileName } })} className="inline-flex shrink-0 items-center justify-center gap-3 rounded-full bg-white px-6 py-3.5 text-[12px] font-bold text-[#111816] transition hover:bg-[#F8FAFC] disabled:cursor-not-allowed disabled:bg-[#94A3B8] disabled:text-white">Open Clinical Review<ArrowRight size={15} /></button></div></section>
            </>
          )}

          <div className="mt-10 flex flex-col justify-between gap-3 border-t border-[#E2E8F0] pt-6 text-[10px] font-medium text-[#64748B] md:flex-row"><div>VisionX Explainability Workspace • API evidence</div><div className="flex items-center gap-2"><ShieldCheck size={13} />No simulated XAI output</div></div>
        </div>
      </div>
    </main>
  );
}
