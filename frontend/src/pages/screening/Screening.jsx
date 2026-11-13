import { useEffect, useRef, useState } from "react";
import { useNavigate, Link } from "react-router-dom";

import {
  Activity,
  ArrowRight,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  CloudOff,
  Eye,
  FileImage,
  ImagePlus,
  Info,
  Network,
  RefreshCcw,
  ScanLine,
  ShieldCheck,
  UploadCloud,
  X,
} from "lucide-react";

import {
  analyzeScreeningImage,
  getScreeningAnalyzeUrl,
  storeAnalysisResult,
} from "../../services/visionxApi";

/* =========================================================
   VISIONX — SCREENING CONSOLE
   Visual language intentionally follows Home.jsx
========================================================= */

const COLORS = {
  background: "#F8FAFC",
  forest: "#0F172A",
  forest2: "#1E293B",
  forest3: "#334155",
  headingMuted: "#0F766E",
  body: "#334155",
  muted: "#64748B",
  border: "#E2E8F0",
  soft: "#F1F5F9",
  accent: "#0F766E",
  success: "#059669",
  darkPanel: "#111816",
  darkPanel2: "#1C2723",
};

const steps = [
  {
    id: 1,
    title: "Image",
    description: "Upload retinal fundus image",
    icon: FileImage,
  },
  {
    id: 2,
    title: "Quality",
    description: "Assess image gradability",
    icon: ScanLine,
  },
  {
    id: 3,
    title: "Analysis",
    description: "Generate retinal findings",
    icon: Activity,
  },
  {
    id: 4,
    title: "Explain",
    description: "Visual evidence with XAI",
    icon: Eye,
  },
  {
    id: 5,
    title: "Review",
    description: "Clinical validation",
    icon: ShieldCheck,
  },
];

/* =========================================================
   STEP INDICATOR
========================================================= */

function StepIndicator({ currentStep }) {
  return (
    <div className="flex items-center">
      {steps.map((step, index) => {
        const Icon = step.icon;

        const complete = currentStep > step.id;
        const active = currentStep === step.id;

        return (
          <div key={step.id} className="flex items-center">
            <div className="flex items-center gap-3">
              <div
                className={[
                  "flex h-10 w-10 items-center justify-center rounded-xl border transition-all duration-300",
                  complete
                    ? "border-[#A7CBB9] bg-[#EAF5EE] text-[#3F7C5E]"
                    : active
                      ? "border-[#111816] bg-[#111816] text-white shadow-[0_7px_20px_rgba(15,23,42,0.15)]"
                      : "border-[#E2E8F0] bg-white text-[#94A3B8]",
                ].join(" ")}
              >
                {complete ? (
                  <CheckCircle2 size={17} strokeWidth={2} />
                ) : (
                  <Icon size={17} strokeWidth={2} />
                )}
              </div>

              <div className="hidden sm:block">
                <div
                  className={`text-[11px] font-bold ${
                    active || complete
                      ? "text-[#334155]"
                      : "text-[#94A3B8]"
                  }`}
                >
                  {step.title}
                </div>

                <div className="mt-0.5 text-[9px] font-medium text-[#94A3B8]">
                  {step.description}
                </div>
              </div>
            </div>

            {index < steps.length - 1 && (
              <div
                className={`mx-4 h-px w-7 md:w-11 ${
                  complete ? "bg-[#A9C7B7]" : "bg-[#E2E8F0]"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

/* =========================================================
   UPLOAD ZONE
========================================================= */

function UploadZone({
  file,
  previewUrl,
  onFile,
  onRemove,
  error,
}) {
  const inputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  const handleFiles = (files) => {
    const selected = files?.[0];

    if (!selected) return;

    const validTypes = [
      "image/jpeg",
      "image/jpg",
      "image/png",
      "image/webp",
    ];

    if (!validTypes.includes(selected.type)) {
      return;
    }

    onFile(selected);
  };

  return (
    <div className="relative">
      {!file ? (
        <div
          onDragEnter={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            setDragActive(false);
          }}
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);
            handleFiles(event.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
          className={[
            "relative flex min-h-[430px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[28px] border-2 border-dashed p-8 text-center transition-all duration-300",
            dragActive
              ? "border-[#0F766E] bg-[#F0FDFA]"
              : "border-[#E2E8F0] bg-[#FBFCFD] hover:border-[#CBD5E1] hover:bg-white",
          ].join(" ")}
        >
          {/* Technical grid */}
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.42]"
            style={{
              backgroundImage:
                "linear-gradient(#E2E8F0 1px, transparent 1px), linear-gradient(90deg, #E2E8F0 1px, transparent 1px)",
              backgroundSize: "52px 52px",
            }}
          />

          {/* Soft center glow */}
          <div className="pointer-events-none absolute left-1/2 top-1/2 h-[310px] w-[310px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-[#F0FDFA] opacity-70 blur-3xl" />

          <input
            ref={inputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            className="hidden"
            onChange={(event) => {
              handleFiles(event.target.files);
              event.target.value = "";
            }}
          />

          <div className="relative z-10">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl border border-[#E2E8F0] bg-white text-[#0F766E] shadow-sm">
              <UploadCloud size={27} strokeWidth={1.7} />
            </div>

            <div className="mt-7 text-[10px] font-bold uppercase tracking-[0.25em] text-[#64748B]">
              Step 01 • Image acquisition
            </div>

            <h3 className="mt-3 text-2xl font-black tracking-tight text-[#0F172A]">
              Upload retinal image
            </h3>

            <p className="mx-auto mt-3 max-w-[500px] text-[14px] leading-6 text-[#475569]">
              Drag and drop a fundus image here, or browse from this
              device to begin the VisionX screening workflow.
            </p>

            <button
              type="button"
              className="mt-7 inline-flex items-center gap-2 rounded-full bg-[#111816] px-6 py-3.5 text-[12px] font-bold text-white shadow-lg transition hover:bg-[#1C2723]"
              onClick={(event) => {
                event.stopPropagation();
                inputRef.current?.click();
              }}
            >
              <ImagePlus size={15} />
              Browse image
              <ArrowRight size={14} />
            </button>

            <div className="mt-5 text-[10px] font-bold uppercase tracking-[0.18em] text-[#94A3B8]">
              JPG • PNG • WEBP
            </div>
          </div>
        </div>
      ) : (
        <div className="relative min-h-[430px] overflow-hidden rounded-[28px] border border-[#E2E8F0] bg-[#020617]">
          <img
            src={previewUrl}
            alt="Uploaded retinal fundus"
            className="h-full min-h-[430px] w-full object-contain"
          />

          {/* Subtle viewer vignette */}
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_42%,rgba(0,0,0,0.32)_100%)]" />

          {/* Image metadata */}
          <div className="absolute left-4 top-4 rounded-2xl border border-white/15 bg-black/55 px-4 py-3 backdrop-blur-md">
            <div className="text-[9px] font-bold uppercase tracking-[0.20em] text-white/45">
              Loaded image
            </div>

            <div className="mt-1 max-w-[260px] truncate text-[12px] font-bold text-white">
              {file.name}
            </div>
          </div>

          {/* Viewer title */}
          <div className="absolute right-4 top-4 rounded-xl border border-white/15 bg-black/50 px-3.5 py-2.5 backdrop-blur-md">
            <div className="text-[9px] font-bold uppercase tracking-[0.16em] text-white/45">
              Fundus viewer
            </div>

            <div className="mt-1 flex items-center gap-2 text-[11px] font-bold text-white">
              <span className="h-1.5 w-1.5 rounded-full bg-[#34D399]" />
              Image loaded
            </div>
          </div>

          {/* Bottom viewer bar */}
          <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between rounded-2xl border border-white/10 bg-black/55 px-4 py-3 backdrop-blur-md">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white/10">
                <FileImage
                  size={15}
                  className="text-white/80"
                />
              </div>

              <div>
                <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-white/45">
                  Ready for quality assessment
                </div>

                <div className="mt-1 text-[11px] font-bold text-white/80">
                  {Math.max(1, Math.round(file.size / 1024))} KB
                </div>
              </div>
            </div>

            <button
              type="button"
              onClick={onRemove}
              className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-white/10 text-white/70 transition hover:bg-white/20 hover:text-white"
              aria-label="Remove image"
            >
              <X size={15} />
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-3 flex items-center gap-2 rounded-xl border border-[#F1D2D0] bg-[#FFF7F7] px-4 py-3 text-[11px] font-semibold text-[#9A5F5B]">
          <Info size={14} />
          {error}
        </div>
      )}
    </div>
  );
}

/* =========================================================
   QUALITY PANEL
========================================================= */

function QualityPanel({ hasImage }) {
  const checks = [
    {
      label: "Field of view",
      state: hasImage ? "Ready" : "Pending",
    },
    {
      label: "Focus",
      state: hasImage ? "Ready" : "Pending",
    },
    {
      label: "Illumination",
      state: hasImage ? "Ready" : "Pending",
    },
    {
      label: "Artifact check",
      state: hasImage ? "Ready" : "Pending",
    },
  ];

  return (
    <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#64748B]">
            Quality gate
          </div>

          <h3 className="mt-2 text-xl font-black tracking-tight text-[#0F172A]">
            Image assessment
          </h3>
        </div>

        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] text-[#0F766E]">
          <ScanLine size={18} />
        </div>
      </div>

      {/* Quality score */}
      <div className="mt-6 rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] p-4">
        <div className="flex items-end justify-between">
          <div>
            <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
              Current status
            </div>

            <div className="mt-2 text-[16px] font-bold text-[#334155]">
              {hasImage ? "Ready for assessment" : "Awaiting image"}
            </div>
          </div>

          <div className="text-right">
            <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
              Quality index
            </div>

            <div className="mt-1 text-[22px] font-black text-[#0F766E]">
              —
            </div>
          </div>
        </div>

        <div className="mt-5 h-2 overflow-hidden rounded-full bg-[#E2E8F0]">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              hasImage ? "w-[18%] bg-[#0F766E]" : "w-0"
            }`}
          />
        </div>

        <div className="mt-2 text-[10px] font-medium text-[#94A3B8]">
          The real quality score will be returned by the VisionX
          image-quality pipeline.
        </div>
      </div>

      {/* Checks */}
      <div className="mt-5 space-y-2">
        {checks.map((item) => (
          <div
            key={item.label}
            className="flex items-center justify-between rounded-xl border border-[#E2E8F0] bg-white px-4 py-3.5"
          >
            <div className="flex items-center gap-3">
              {hasImage ? (
                <CircleDot
                  size={15}
                  className="text-[#0F766E]"
                />
              ) : (
                <CircleDot
                  size={15}
                  className="text-[#CBD5E1]"
                />
              )}

              <span className="text-[12px] font-bold text-[#475569]">
                {item.label}
              </span>
            </div>

            <span
              className={`text-[10px] font-bold uppercase tracking-[0.12em] ${
                hasImage
                  ? "text-[#0F766E]"
                  : "text-[#94A3B8]"
              }`}
            >
              {item.state}
            </span>
          </div>
        ))}
      </div>

      {/* Info */}
      <div className="mt-5 flex items-start gap-3 rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] px-4 py-3.5">
        <Info
          size={15}
          className="mt-0.5 shrink-0 text-[#0F766E]"
        />

        <p className="text-[11px] leading-5 text-[#64748B]">
          Image quality is assessed before retinal analysis
          and downstream clinical review.
        </p>
      </div>
    </div>
  );
}

/* =========================================================
   CASE SUMMARY
========================================================= */

function InfoRow({ label, value, icon }) {
  return (
    <div className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3.5">
      <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
        {label}
      </span>

      <div className="flex items-center gap-2 text-[11px] font-bold text-[#334155]">
        {icon}
        {value}
      </div>
    </div>
  );
}

function CaseSummary({ file }) {
  return (
    <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
      <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#64748B]">
        Image metadata
      </div>

      <div className="mt-5 space-y-2.5">
        <InfoRow label="File" value={file?.name} />
        <InfoRow label="Type" value={file?.type} />
        <InfoRow
          label="Size"
          value={
            file
              ? `${Math.max(1, Math.round(file.size / 1024))} KB`
              : "—"
          }
        />
        <InfoRow label="Case ID" value="Assigned by inference service" />
      </div>

      <div className="mt-5 border-t border-[#E2E8F0] pt-5">
        <div className="flex items-start gap-2.5 text-[10px] leading-5 text-[#64748B]">
          <ShieldCheck size={14} className="mt-0.5 shrink-0 text-[#059669]" />
          Clinical and case metadata will be displayed only when returned by the connected service.
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   WORKFLOW STATUS
========================================================= */

function WorkflowStatus({ currentStep, hasImage }) {
  const items = [
    {
      number: "01",
      title: "Acquisition",
      text: hasImage ? "Image received" : "Awaiting image",
      state: hasImage ? "complete" : "current",
    },
    {
      number: "02",
      title: "Quality gate",
      text: currentStep >= 2 ? "Ready for assessment" : "Awaiting image",
      state: currentStep >= 2 ? "current" : "pending",
    },
    {
      number: "03",
      title: "Retinal analysis",
      text: "Next workspace",
      state: "pending",
    },
    {
      number: "04",
      title: "Clinical review",
      text: "Human validation",
      state: "pending",
    },
  ];

  return (
    <section className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {items.map((item) => (
        <div
          key={item.number}
          className="rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm"
        >
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-bold tracking-[0.18em] text-[#94A3B8]">
              {item.number}
            </span>

            {item.state === "complete" ? (
              <CheckCircle2
                size={15}
                className="text-[#059669]"
              />
            ) : item.state === "current" ? (
              <span className="h-2 w-2 rounded-full bg-[#0F766E]" />
            ) : (
              <span className="h-2 w-2 rounded-full bg-[#CBD5E1]" />
            )}
          </div>

          <div className="mt-8 text-[13px] font-bold text-[#334155]">
            {item.title}
          </div>

          <div className="mt-1 text-[10px] font-medium text-[#64748B]">
            {item.text}
          </div>
        </div>
      ))}
    </section>
  );
}

/* =========================================================
   MAIN SCREENING PAGE
========================================================= */

export default function Screening() {
  const navigate = useNavigate();

  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [currentStep, setCurrentStep] = useState(1);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState("");
  const endpointConfigured = Boolean(getScreeningAnalyzeUrl());

  /* -------------------------------------------------------
     Safe preview URL lifecycle
  ------------------------------------------------------- */

  useEffect(() => {
    if (!file) {
      setPreviewUrl("");
      return undefined;
    }

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  /* -------------------------------------------------------
     File handling
  ------------------------------------------------------- */

  const handleFile = (selectedFile, validationError = "") => {
    if (validationError) {
      setError(validationError);
      return;
    }

    if (!selectedFile) return;

    const validTypes = [
      "image/jpeg",
      "image/jpg",
      "image/png",
      "image/webp",
    ];

    if (!validTypes.includes(selectedFile.type)) {
      setError("Please upload a valid JPG, PNG or WEBP retinal image.");
      return;
    }

    setError("");
    setFile(selectedFile);
    setCurrentStep(2);
  };

  const handleRemove = () => {
    setFile(null);
    setPreviewUrl("");
    setCurrentStep(1);
    setError("");
  };

  /* -------------------------------------------------------
     Real inference request
  ------------------------------------------------------- */

  const handleAnalyze = async () => {
    if (!file || isProcessing) return;

    if (!getScreeningAnalyzeUrl()) {
      setError(
        "Screening inference endpoint is not configured. Add VITE_SCREENING_ANALYZE_URL to the frontend environment."
      );
      return;
    }

    setIsProcessing(true);
    setError("");

    try {
      const result = await analyzeScreeningImage(file);

      storeAnalysisResult(result);
      setCurrentStep(3);

      navigate("/screening/analysis", {
        state: {
          analysisResult: result,
          file,
          fileName: file.name,
        },
      });
    } catch (requestError) {
      setError(
        requestError?.message ||
          "The screening inference request failed."
      );
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <main
      className="min-h-screen overflow-hidden text-[#0F172A]"
      style={{
        backgroundColor: COLORS.background,
      }}
    >
      {/* =====================================================
          BACKGROUND GRID
      ===================================================== */}

      <div
        className="pointer-events-none fixed inset-0 opacity-[0.45]"
        style={{
          backgroundImage:
            "linear-gradient(#E2E8F0 1px, transparent 1px), linear-gradient(90deg, #E2E8F0 1px, transparent 1px)",
          backgroundSize: "72px 72px",
        }}
      />

      <div className="pointer-events-none fixed right-[-220px] top-[-180px] h-[700px] w-[700px] rounded-full bg-[#E0F2FE] blur-[140px]" />

      <div className="relative z-10">
        {/* ===================================================
            HEADER
        =================================================== */}

        <header className="border-b border-[#E2E8F0] bg-white/95 backdrop-blur-md">
          <div className="mx-auto flex max-w-[1460px] items-center justify-between px-6 py-5 md:px-10 lg:px-12">
            {/* Brand */}

            <Link
              to="/"
              className="group flex items-center gap-3"
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#0F766E] shadow-sm transition group-hover:border-[#CBD5E1]">
                <Eye size={21} strokeWidth={2} />
              </div>

              <div>
                <div className="text-[17px] font-black tracking-[0.29em] text-[#0F172A]">
                  VISIONX
                </div>

                <div className="mt-0.5 text-[9px] font-bold uppercase tracking-[0.22em] text-[#64748B]">
                  Screening Console
                </div>
              </div>
            </Link>

            {/* Steps */}

            <div className="hidden xl:block">
              <StepIndicator currentStep={currentStep} />
            </div>

            {/* Right */}

            <div className="flex items-center gap-3">
              <div
                className={`hidden items-center gap-2 rounded-full border px-3.5 py-2 text-[9px] font-bold uppercase tracking-[0.14em] sm:flex ${
                  endpointConfigured
                    ? "border-[#D1FAE5] bg-[#ECFDF5] text-[#047857]"
                    : "border-[#E2E8F0] bg-[#F8FAFC] text-[#64748B]"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    endpointConfigured
                      ? "bg-[#059669]"
                      : "bg-[#94A3B8]"
                  }`}
                />
                {endpointConfigured
                  ? "Inference endpoint configured"
                  : "Inference endpoint not configured"}
              </div>

              <Link
                to="/"
                className="rounded-full border border-[#E2E8F0] bg-white px-4 py-2.5 text-[10px] font-bold text-[#64748B] transition hover:border-[#CBD5E1] hover:text-[#334155]"
              >
                Exit
              </Link>
            </div>
          </div>
        </header>

        {/* ===================================================
            MAIN PAGE
        =================================================== */}

        <div className="mx-auto max-w-[1460px] px-6 py-10 md:px-10 lg:px-12">
          {/* Mobile steps */}

          <div className="mb-7 overflow-x-auto xl:hidden">
            <StepIndicator currentStep={currentStep} />
          </div>

          {/* =================================================
              PAGE HEADING
          ================================================= */}

          <section className="flex flex-col justify-between gap-7 lg:flex-row lg:items-end">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">
                New screening case
              </div>

              <h1 className="mt-4 text-[42px] font-black leading-[1] tracking-tight text-[#0F172A] md:text-[52px]">
                Start a retinal
                <span className="block text-[#0F766E]">
                  screening.
                </span>
              </h1>

              <p className="mt-6 max-w-[700px] text-[16px] font-medium leading-relaxed text-[#475569]">
                Upload a fundus image and move through image
                quality assessment, retinal analysis, visual
                evidence and clinical review in one connected
                workflow.
              </p>
            </div>

            <div className="flex w-fit items-center gap-3 rounded-2xl border border-[#E2E8F0] bg-white px-5 py-4 shadow-sm">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
                <Network size={16} />
              </div>

              <div>
                <div className="text-[9px] font-bold uppercase tracking-[0.18em] text-[#94A3B8]">
                  Session
                </div>

                <div className="mt-1 text-[12px] font-bold text-[#334155]">
                  Local screening workspace
                </div>
              </div>
            </div>
          </section>

          {/* =================================================
              WORKFLOW INTRO
          ================================================= */}

          <section className="mt-10 grid gap-4 md:grid-cols-4">
            {[
              ["01", "Image quality", "Quality before inference"],
              ["02", "Retinal analysis", "Structured findings"],
              ["03", "Visual evidence", "Grad-CAM + lesions"],
              ["04", "Clinical review", "Human validation"],
            ].map(([number, title, text], index) => (
              <div
                key={title}
                className={`rounded-2xl border border-[#E2E8F0] bg-white px-5 py-5 shadow-sm ${
                  index === 0
                    ? "border-t-2 border-t-[#0F766E]"
                    : ""
                }`}
              >
                <div className="text-[10px] font-bold tracking-[0.22em] text-[#94A3B8]">
                  {number}
                </div>

                <div className="mt-3 text-[13px] font-bold text-[#0F172A]">
                  {title}
                </div>

                <div className="mt-1 text-[10px] font-medium text-[#64748B]">
                  {text}
                </div>
              </div>
            ))}
          </section>

          {/* =================================================
              MAIN WORKSPACE
          ================================================= */}

          <div className="mt-8 grid gap-6 xl:grid-cols-[1.38fr_0.62fr]">
            {/* LEFT */}

            <section className="rounded-[30px] border border-[#E2E8F0] bg-white p-4 shadow-sm md:p-5">
              <div className="flex flex-col justify-between gap-4 px-1 pb-5 sm:flex-row sm:items-center">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#64748B]">
                    Acquisition workspace
                  </div>

                  <h2 className="mt-2 text-2xl font-black tracking-tight text-[#0F172A]">
                    Retinal image
                  </h2>
                </div>

                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.16em] text-[#64748B]">
                  <CloudOff size={14} />
                  Offline capable
                </div>
              </div>

              <UploadZone
                file={file}
                previewUrl={previewUrl}
                onFile={handleFile}
                onRemove={handleRemove}
                error={error}
              />

              {/* -------------------------------------------------
                  ACTION BAR
              ------------------------------------------------- */}

              <div className="mt-4 flex flex-col justify-between gap-5 rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] p-5 md:flex-row md:items-center">
                <div className="flex items-start gap-3">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#0F766E] shadow-sm">
                    <Info size={16} />
                  </div>

                  <div>
                    <div className="text-[12px] font-bold text-[#334155]">
                      Before retinal inference
                    </div>

                    <p className="mt-1 max-w-[500px] text-[10px] font-medium leading-5 text-[#64748B]">
                      VisionX first validates image adequacy.
                      Only an acceptable image proceeds to
                      disease analysis and evidence generation.
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  disabled={!file || isProcessing || !endpointConfigured}
                  onClick={handleAnalyze}
                  className="inline-flex items-center justify-center gap-3 rounded-full bg-[#111816] px-6 py-3.5 text-[12px] font-bold text-white shadow-lg transition hover:bg-[#1C2723] disabled:cursor-not-allowed disabled:bg-[#94A3B8]"
                >
                  {isProcessing ? (
                    <>
                      <RefreshCcw
                        size={14}
                        className="animate-spin"
                      />
                      Preparing analysis
                    </>
                  ) : (
                    <>
                      Analyze image
                      <ArrowRight size={15} />
                    </>
                  )}
                </button>
              </div>
            </section>

            {/* RIGHT */}

            <aside className="space-y-6">
              <QualityPanel hasImage={Boolean(file)} />

              <CaseSummary file={file} />
            </aside>
          </div>

          {/* =================================================
              SECONDARY STATUS
          ================================================= */}

          <WorkflowStatus
            currentStep={currentStep}
            hasImage={Boolean(file)}
          />

          {/* =================================================
              INFORMATION STRIP
          ================================================= */}

          <section className="mt-8 grid gap-5 lg:grid-cols-3">
            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <ScanLine size={19} />
              </div>

              <div className="mt-6 text-[14px] font-bold text-[#0F172A]">
                Quality first
              </div>

              <p className="mt-2 text-[12px] leading-5 text-[#64748B]">
                Focus, illumination, field of view and artifacts
                are checked before downstream interpretation.
              </p>
            </div>

            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <Activity size={19} />
              </div>

              <div className="mt-6 text-[14px] font-bold text-[#0F172A]">
                Retinal analysis
              </div>

              <p className="mt-2 text-[12px] leading-5 text-[#64748B]">
                The next workspace generates retinal findings and disease
                grading before the dedicated explainability stage.
              </p>
            </div>

            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <ShieldCheck size={19} />
              </div>

              <div className="mt-6 text-[14px] font-bold text-[#0F172A]">
                Human-in-the-loop
              </div>

              <p className="mt-2 text-[12px] leading-5 text-[#64748B]">
                Final screening decisions remain inside a
                clinically reviewable workflow.
              </p>
            </div>
          </section>

          {/* =================================================
              FOOTER NOTE
          ================================================= */}

          <div className="mt-10 flex flex-col justify-between gap-3 border-t border-[#E2E8F0] pt-6 text-[10px] font-medium text-[#64748B] md:flex-row">
            <div>
              VisionX Screening Console • Retinal image
              acquisition
            </div>

            <div className="flex items-center gap-2">
              <ChevronRight size={13} />
              Next: retinal analysis workspace
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}