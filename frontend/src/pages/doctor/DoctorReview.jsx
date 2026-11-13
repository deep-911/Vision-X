import { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams, Link } from "react-router-dom";

import {
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  CircleDot,
  Eye,
  FileCheck2,
  Info,
  MessageSquare,
  Network,
  ScanLine,
  ShieldCheck,
  Stethoscope,
  UserRound,
} from "lucide-react";

/* =========================================================
   VISIONX — CLINICAL REVIEW WORKSPACE
   Same visual system as Home.jsx
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
   REVIEW WORKFLOW
========================================================= */

function StepIndicator() {
  const steps = [
    ["Image", true],
    ["Quality", true],
    ["Analysis", true],
    ["Explain", true],
    ["Review", true],
  ];

  return (
    <div className="hidden items-center xl:flex">
      {steps.map(([label, complete], index) => (
        <div key={label} className="flex items-center">
          <div className="flex items-center gap-2.5">
            <div
              className={`flex h-9 w-9 items-center justify-center rounded-xl border ${
                label === "Review"
                  ? "border-[#111816] bg-[#111816] text-white"
                  : complete
                    ? "border-[#A7CBB9] bg-[#EAF5EE] text-[#3F7C5E]"
                    : "border-[#E2E8F0] bg-white text-[#94A3B8]"
              }`}
            >
              {label !== "Review" ? (
                <CheckCircle2 size={16} />
              ) : (
                <span className="text-[10px] font-black">
                  5
                </span>
              )}
            </div>

            <span
              className={`text-[11px] font-bold ${
                label === "Review"
                  ? "text-[#0F172A]"
                  : "text-[#334155]"
              }`}
            >
              {label}
            </span>
          </div>

          {index < steps.length - 1 && (
            <div className="mx-4 h-px w-8 bg-[#A9C7B7]" />
          )}
        </div>
      ))}
    </div>
  );
}

/* =========================================================
   IMAGE PANEL
========================================================= */

function ReviewImage({ previewUrl, fileName }) {
  return (
    <div className="relative min-h-[520px] overflow-hidden rounded-[28px] bg-[#020617]">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.03),transparent_58%)]" />

      {previewUrl ? (
        <img
          src={previewUrl}
          alt="Retinal image for clinical review"
          className="absolute inset-0 h-full w-full object-contain"
        />
      ) : (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
            <Eye
              size={40}
              className="mx-auto text-white/20"
            />

            <div className="mt-4 text-[11px] font-bold text-white/45">
              Retinal image unavailable
            </div>
          </div>
        </div>
      )}

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_40%,rgba(0,0,0,0.42)_100%)]" />

      {/* Viewer information */}

      <div className="absolute left-4 top-4 rounded-2xl border border-white/15 bg-black/55 px-4 py-3 backdrop-blur-md">
        <div className="text-[9px] font-bold uppercase tracking-[0.20em] text-white/45">
          Clinical review
        </div>

        <div className="mt-1 max-w-[260px] truncate text-[12px] font-bold text-white">
          {fileName || "Case VX-NEW"}
        </div>
      </div>

      <div className="absolute right-4 top-4 rounded-full border border-[#2B3B35] bg-[#1C2723]/90 px-4 py-2 text-[9px] font-bold uppercase tracking-[0.14em] text-[#A9BEB4]">
        Reviewer workspace
      </div>

      {/* Bottom controls */}

      <div className="absolute bottom-4 left-4 flex gap-2">
        {["Original", "Evidence"].map((item, index) => (
          <button
            key={item}
            type="button"
            className={`rounded-lg border px-3 py-2 text-[10px] font-bold ${
              index === 0
                ? "border-white/30 bg-white/15 text-white"
                : "border-white/15 bg-black/45 text-white/55"
            }`}
          >
            {item}
          </button>
        ))}
      </div>

      <div className="absolute bottom-4 right-4 rounded-lg border border-white/15 bg-black/50 px-3 py-2 text-[9px] font-bold uppercase tracking-[0.14em] text-white/50">
        RGB • fundus
      </div>
    </div>
  );
}

/* =========================================================
   RESULT SUMMARY
========================================================= */

function FindingCard({
  label,
  value,
  icon: Icon,
  tone = "neutral",
}) {
  const toneClasses = {
    neutral: {
      box: "bg-[#F8FAFC]",
      icon: "bg-[#F1F5F9] text-[#0F766E]",
      value: "text-[#0F172A]",
    },
    success: {
      box: "bg-[#F3FAF6]",
      icon: "bg-[#ECFDF5] text-[#059669]",
      value: "text-[#0F172A]",
    },
    warning: {
      box: "bg-[#FFFBEB]",
      icon: "bg-[#FEF3C7] text-[#B7791F]",
      value: "text-[#0F172A]",
    },
  };

  const style = toneClasses[tone];

  return (
    <div className={`rounded-2xl border border-[#E2E8F0] p-5 ${style.box}`}>
      <div className="flex items-start justify-between">
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-xl ${style.icon}`}
        >
          <Icon size={17} />
        </div>

        <CircleDot
          size={14}
          className="text-[#CBD5E1]"
        />
      </div>

      <div className="mt-5 text-[10px] font-bold uppercase tracking-[0.16em] text-[#64748B]">
        {label}
      </div>

      <div
        className={`mt-2 text-[21px] font-black ${style.value}`}
      >
        {value}
      </div>
    </div>
  );
}

/* =========================================================
   REVIEW CHECKLIST
========================================================= */

function ReviewChecklist() {
  const items = [
    "Image quality reviewed",
    "Retinal findings reviewed",
    "Visual evidence reviewed",
    "Model confidence considered",
  ];

  return (
    <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
      <div className="flex items-start justify-between">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-[0.23em] text-[#0F766E]">
            Review checklist
          </div>

          <h3 className="mt-2 text-xl font-black text-[#0F172A]">
            Clinical validation
          </h3>
        </div>

        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
          <Stethoscope size={18} />
        </div>
      </div>

      <div className="mt-5 space-y-2.5">
        {items.map((item) => (
          <div
            key={item}
            className="flex items-center gap-3 rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] px-4 py-3.5"
          >
            <CheckCircle2
              size={16}
              className="text-[#059669]"
            />

            <span className="text-[11px] font-bold text-[#334155]">
              {item}
            </span>
          </div>
        ))}
      </div>

      <div className="mt-5 flex items-start gap-3 rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] p-4">
        <Info
          size={15}
          className="mt-0.5 shrink-0 text-[#0F766E]"
        />

        <p className="text-[10px] leading-5 text-[#64748B]">
          This workspace is for human validation of the
          screening output and supporting evidence.
        </p>
      </div>
    </div>
  );
}

/* =========================================================
   REVIEW DECISION
========================================================= */

function ReviewDecision({ decision, setDecision }) {
  const options = [
    {
      id: "validated",
      title: "Validate finding",
      text: "Accept the model-supported screening interpretation.",
      icon: CheckCircle2,
    },
    {
      id: "review",
      title: "Needs further review",
      text: "Keep the case open for additional clinical assessment.",
      icon: Eye,
    },
    {
      id: "refer",
      title: "Refer case",
      text: "Move the case into the referral workflow.",
      icon: ArrowRight,
    },
  ];

  return (
    <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
      <div>
        <div className="text-[10px] font-bold uppercase tracking-[0.23em] text-[#0F766E]">
          Case disposition
        </div>

        <h3 className="mt-2 text-xl font-black text-[#0F172A]">
          Reviewer decision
        </h3>

        <p className="mt-2 text-[11px] leading-5 text-[#64748B]">
          Select the workflow state after reviewing the case
          evidence.
        </p>
      </div>

      <div className="mt-5 space-y-3">
        {options.map((option) => {
          const Icon = option.icon;
          const active = decision === option.id;

          return (
            <button
              key={option.id}
              type="button"
              onClick={() => setDecision(option.id)}
              className={`w-full rounded-2xl border p-4 text-left transition ${
                active
                  ? "border-[#A9D5C4] bg-[#F3FAF6]"
                  : "border-[#E2E8F0] bg-white hover:bg-[#F8FAFC]"
              }`}
            >
              <div className="flex items-start gap-3">
                <div
                  className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${
                    active
                      ? "bg-[#0F766E] text-white"
                      : "bg-[#F1F5F9] text-[#64748B]"
                  }`}
                >
                  <Icon size={17} />
                </div>

                <div className="flex-1">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-[12px] font-bold text-[#0F172A]">
                      {option.title}
                    </div>

                    {active && (
                      <CheckCircle2
                        size={15}
                        className="text-[#059669]"
                      />
                    )}
                  </div>

                  <div className="mt-1 text-[10px] leading-5 text-[#64748B]">
                    {option.text}
                  </div>
                </div>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* =========================================================
   CASE INFORMATION
========================================================= */

function CaseInformation({ caseId }) {
  return (
    <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
      <div className="text-[10px] font-bold uppercase tracking-[0.23em] text-[#64748B]">
        Case information
      </div>

      <div className="mt-5 space-y-2.5">
        <div className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3.5">
          <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
            Case ID
          </span>

          <span className="text-[11px] font-bold text-[#334155]">
            {caseId || "VX-NEW"}
          </span>
        </div>

        <div className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3.5">
          <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
            Eye
          </span>

          <span className="text-[11px] font-bold text-[#334155]">
            Not specified
          </span>
        </div>

        <div className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3.5">
          <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
            Capture device
          </span>

          <span className="text-[11px] font-bold text-[#334155]">
            PHC-CAM-04
          </span>
        </div>

        <div className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3.5">
          <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
            Connectivity
          </span>

          <div className="flex items-center gap-2 text-[11px] font-bold text-[#334155]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#059669]" />
            Online
          </div>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   MAIN PAGE
========================================================= */

export default function DoctorReview() {
  const navigate = useNavigate();
  const location = useLocation();
  const { id } = useParams();

  const file = location.state?.file || null;
  const fileName =
    location.state?.fileName || file?.name || "";
  const caseId = id || location.state?.caseId || "VX-NEW";

  const analysisResult =
    location.state?.analysisResult ||
    (() => {
      try {
        const stored = sessionStorage.getItem("visionx.analysis.result");
        return stored ? JSON.parse(stored) : null;
      } catch {
        return null;
      }
    })();

  const displayGrade =
    analysisResult?.grade ??
    analysisResult?.icdrGrade ??
    analysisResult?.matlabResult?.grade ??
    "Pending";

  const displayConfidence =
    analysisResult?.confidence ??
    analysisResult?.matlabResult?.confidence ??
    "Pending";

  const [previewUrl, setPreviewUrl] = useState("");
  const [decision, setDecision] = useState("");
  const [reviewerNote, setReviewerNote] = useState(() => {
    try {
      return (
        localStorage.getItem(`visionx.review.draft.${caseId}`) || ""
      );
    } catch {
      return "";
    }
  });
  const [saveStatus, setSaveStatus] = useState("Not saved yet");

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

  useEffect(() => {
    try {
      localStorage.setItem(
        `visionx.review.draft.${caseId}`,
        reviewerNote
      );
      setSaveStatus(reviewerNote.trim() ? "Saved locally" : "Not saved yet");
    } catch {
      setSaveStatus("Save unavailable");
    }
  }, [caseId, reviewerNote]);

  const handleBack = () => {
    navigate("/screening/explainability", {
      state: {
        file,
        fileName,
      },
    });
  };

  const handleComplete = () => {
    if (!decision) return;

    const reviewRecord = {
      caseId,
      decision,
      reviewerNote: reviewerNote.trim(),
      reviewer: "Clinical workspace",
      timestamp: new Date().toISOString(),
    };

    try {
      localStorage.setItem(
        `visionx.review.${caseId}`,
        JSON.stringify(reviewRecord)
      );
      localStorage.setItem(
        `visionx.review.draft.${caseId}`,
        reviewerNote
      );
      setSaveStatus("Review saved locally");
    } catch {
      setSaveStatus("Save unavailable");
    }

    navigate("/doctor", {
      state: reviewRecord,
    });
  };

  return (
    <main className="min-h-screen overflow-hidden bg-[#F8FAFC] text-[#0F172A]">
      {/* =====================================================
          BACKGROUND GRID
      ===================================================== */}

      <div
        className="pointer-events-none fixed inset-0 opacity-[0.38]"
        style={{
          backgroundImage:
            "linear-gradient(#E2E8F0 1px, transparent 1px), linear-gradient(90deg, #E2E8F0 1px, transparent 1px)",
          backgroundSize: "72px 72px",
        }}
      />

      <div className="relative z-10">
        {/* =================================================
            HEADER
        ================================================= */}

        <header className="border-b border-[#E2E8F0] bg-white/95 backdrop-blur-md">
          <div className="mx-auto flex max-w-[1460px] items-center justify-between px-6 py-5 md:px-10 lg:px-12">
            <Link
              to="/"
              className="flex items-center gap-3"
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#0F766E] shadow-sm">
                <Stethoscope size={20} />
              </div>

              <div>
                <div className="text-[17px] font-black tracking-[0.29em] text-[#0F172A]">
                  VISIONX
                </div>

                <div className="mt-0.5 text-[9px] font-bold uppercase tracking-[0.22em] text-[#64748B]">
                  Clinical Review
                </div>
              </div>
            </Link>

            <StepIndicator />

            <button
              type="button"
              onClick={handleBack}
              className="inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-2.5 text-[10px] font-bold text-[#64748B] transition hover:border-[#CBD5E1] hover:text-[#334155]"
            >
              <ArrowLeft size={13} />
              Back
            </button>
          </div>
        </header>

        {/* =================================================
            CONTENT
        ================================================= */}

        <div className="mx-auto max-w-[1460px] px-6 py-10 md:px-10 lg:px-12">
          {/* Heading */}

          <section className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">
                Step 05 • Human-in-the-loop
              </div>

              <h1 className="mt-4 text-[42px] font-black leading-none tracking-tight text-[#0F172A] md:text-[54px]">
                Review the
                <span className="block text-[#0F766E]">
                  screening case.
                </span>
              </h1>

              <p className="mt-6 max-w-[720px] text-[16px] font-medium leading-relaxed text-[#475569]">
                Inspect the retinal image, model output and
                supporting evidence before recording the final
                workflow state.
              </p>
            </div>

            <div className="flex w-fit items-center gap-3 rounded-2xl border border-[#E2E8F0] bg-white px-5 py-4 shadow-sm">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
                <UserRound size={16} />
              </div>

              <div>
                <div className="text-[9px] font-bold uppercase tracking-[0.17em] text-[#94A3B8]">
                  Reviewer
                </div>

                <div className="mt-1 text-[12px] font-bold text-[#334155]">
                  Clinical workspace
                </div>
              </div>
            </div>
          </section>

          {/* =================================================
              CASE STRIP
          ================================================= */}

          <section className="mt-8 flex flex-col justify-between gap-4 rounded-2xl border border-[#E2E8F0] bg-white px-5 py-4 shadow-sm md:flex-row md:items-center">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
                <FileCheck2 size={17} />
              </div>

              <div>
                <div className="text-[9px] font-bold uppercase tracking-[0.17em] text-[#94A3B8]">
                  Screening case
                </div>

                <div className="mt-1 text-[12px] font-bold text-[#334155]">
                  {caseId}
                </div>
              </div>
            </div>

            <div className="flex flex-wrap gap-3">
              <span className="rounded-full border border-[#E2E8F0] bg-[#F8FAFC] px-3 py-2 text-[9px] font-bold uppercase tracking-[0.12em] text-[#64748B]">
                Analysis complete
              </span>

              <span className="rounded-full border border-[#D1FAE5] bg-[#ECFDF5] px-3 py-2 text-[9px] font-bold uppercase tracking-[0.12em] text-[#047857]">
                Evidence available
              </span>

              <span className="rounded-full border border-[#E2E8F0] bg-white px-3 py-2 text-[9px] font-bold uppercase tracking-[0.12em] text-[#64748B]">
                Review pending
              </span>
            </div>
          </section>

          {/* =================================================
              MAIN GRID
          ================================================= */}

          <div className="mt-6 grid gap-6 xl:grid-cols-[1.35fr_0.65fr]">
            {/* LEFT */}

            <section className="rounded-[30px] border border-[#E2E8F0] bg-white p-4 shadow-sm md:p-5">
              <div className="flex items-center justify-between px-1 pb-5">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#64748B]">
                    Retinal examination
                  </div>

                  <h2 className="mt-2 text-xl font-black text-[#0F172A]">
                    Review image
                  </h2>
                </div>

                <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.13em] text-[#64748B]">
                  <ScanLine size={14} />
                  Fundus viewer
                </div>
              </div>

              <ReviewImage
                previewUrl={previewUrl}
                fileName={fileName}
              />

              {/* Findings */}

              <div className="mt-4 grid gap-3 md:grid-cols-3">
                <FindingCard
                  label="ICDR grade"
                  value={displayGrade}
                  icon={Eye}
                />

                <FindingCard
                  label="Confidence"
                  value={
                    displayConfidence === "Pending"
                      ? "Pending"
                      : `${(Number(displayConfidence) <= 1 ? Number(displayConfidence) * 100 : Number(displayConfidence)).toFixed(1)}%`
                  }
                  icon={CircleDot}
                />

                <FindingCard
                  label="Review status"
                  value="Open"
                  icon={ShieldCheck}
                  tone="warning"
                />
              </div>
            </section>

            {/* RIGHT */}

            <aside className="space-y-6">
              <ReviewChecklist />

              <CaseInformation caseId={caseId} />

              {/* Evidence reminder */}

              <div className="overflow-hidden rounded-[28px] bg-[#111816] p-5 shadow-xl">
                <div className="rounded-[22px] border border-[#2B3B35] bg-[#1C2723] p-5">
                  <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.20em] text-[#80958B]">
                    <Eye size={14} />
                    Evidence layer
                  </div>

                  <h3 className="mt-3 text-[18px] font-black text-white">
                    Inspect before deciding.
                  </h3>

                  <p className="mt-3 text-[11px] leading-5 text-[#A9BEB4]">
                    Grad-CAM attention, lesion evidence and
                    model confidence should be considered together
                    during clinical validation.
                  </p>

                  <button
                    type="button"
                    onClick={handleBack}
                    className="mt-5 inline-flex items-center gap-2 rounded-full border border-[#2B3B35] bg-[#16201D] px-4 py-2.5 text-[10px] font-bold text-[#DCE8E1] transition hover:bg-[#25312C]"
                  >
                    Re-open evidence
                    <ArrowRight size={13} />
                  </button>
                </div>
              </div>
            </aside>
          </div>

          {/* =================================================
              DECISION
          ================================================= */}

          <div className="mt-7 grid gap-6 lg:grid-cols-[1fr_0.72fr]">
            <ReviewDecision
              decision={decision}
              setDecision={setDecision}
            />

            <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.23em] text-[#0F766E]">
                <MessageSquare size={14} />
                Reviewer note
              </div>

              <h3 className="mt-3 text-xl font-black text-[#0F172A]">
                Add context.
              </h3>

              <p className="mt-2 text-[11px] leading-5 text-[#64748B]">
                Clinical notes can be attached to the case for
                auditability and downstream referral.
              </p>

              <textarea
                rows={6}
                value={reviewerNote}
                onChange={(event) => setReviewerNote(event.target.value)}
                placeholder="Enter reviewer note..."
                className="mt-5 w-full resize-none rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] px-4 py-3.5 text-[12px] font-medium text-[#334155] outline-none transition placeholder:text-[#94A3B8] focus:border-[#A9D5C4] focus:bg-white"
              />

              <div className="mt-3 flex items-center justify-between gap-3 text-[9px] font-medium text-[#94A3B8]">
                <span>
                  Notes are saved in this browser and attached to the
                  screening case for this demo session.
                </span>
                <span className="shrink-0 font-bold text-[#059669]">
                  {saveStatus}
                </span>
              </div>
            </div>
          </div>

          {/* =================================================
              FINAL ACTION
          ================================================= */}

          <section className="mt-7 overflow-hidden rounded-[30px] bg-[#111816] px-6 py-8 shadow-2xl md:px-8">
            <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-center">
              <div>
                <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#80958B]">
                  Final workflow action
                </div>

                <h2 className="mt-3 text-2xl font-black text-white">
                  Record the
                  <span className="text-[#80958B]">
                    {" "}clinical state.
                  </span>
                </h2>

                <p className="mt-3 max-w-[650px] text-[12px] leading-6 text-[#A9BEB4]">
                  The selected disposition can later trigger the
                  appropriate case queue or referral workflow.
                </p>
              </div>

              <button
                type="button"
                disabled={!decision}
                onClick={handleComplete}
                className="inline-flex shrink-0 items-center justify-center gap-3 rounded-full bg-white px-6 py-3.5 text-[12px] font-bold text-[#111816] transition hover:bg-[#F8FAFC] disabled:cursor-not-allowed disabled:bg-[#94A3B8] disabled:text-white"
              >
                Complete review
                <ArrowRight size={15} />
              </button>
            </div>
          </section>

          {/* =================================================
              FOOTER
          ================================================= */}

          <div className="mt-10 flex flex-col justify-between gap-3 border-t border-[#E2E8F0] pt-6 text-[10px] font-medium text-[#64748B] md:flex-row">
            <div>
              VisionX Clinical Review • Human-in-the-loop
            </div>

            <div className="flex items-center gap-2">
              <Network size={13} />
              Reviewable screening workflow
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}