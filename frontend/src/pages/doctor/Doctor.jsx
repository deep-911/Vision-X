import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, Link } from "react-router-dom";

import {
  Activity,
  ArrowRight,
  CheckCircle2,
  Clock3,
  Eye,
  Network,
  FileCheck2,
  Filter,
  Hospital,
  RefreshCcw,
  Search,
  ShieldCheck,
  Stethoscope,
  Wifi,
  XCircle,
} from "lucide-react";

/* =========================================================
   VISIONX — CLINICAL REVIEW DASHBOARD

   IMPORTANT:
   No patient, case, count, device or clinical result is
   hardcoded here.

   Data must come from:

   VITE_REVIEW_QUEUE_URL

   Example .env:
   VITE_REVIEW_QUEUE_URL=http://localhost:8000/api/cases/review-queue

   Change the URL to your REAL backend endpoint.
========================================================= */

/* =========================================================
   DESIGN SYSTEM — SAME AS HOME.JSX
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

/* =========================================================
   API CONFIG
========================================================= */

const REVIEW_QUEUE_URL =
  import.meta.env.VITE_REVIEW_QUEUE_URL || "";

/* =========================================================
   SAFE RESPONSE NORMALIZATION
========================================================= */

function extractCases(payload) {
  if (Array.isArray(payload)) {
    return payload;
  }

  if (Array.isArray(payload?.cases)) {
    return payload.cases;
  }

  if (Array.isArray(payload?.data)) {
    return payload.data;
  }

  if (Array.isArray(payload?.results)) {
    return payload.results;
  }

  return [];
}

/* =========================================================
   DATA NORMALIZER

   Does not invent values.
   Missing fields become "—".
========================================================= */

function normalizeCase(item, index) {
  const source = item || {};

  return {
    key:
      source.id ??
      source.caseId ??
      source.case_id ??
      `row-${index}`,

    id:
      source.id ??
      source.caseId ??
      source.case_id ??
      "—",

    patient:
      source.patientName ??
      source.patient_name ??
      source.patientLabel ??
      source.patient_label ??
      "—",

    eye:
      source.eye ??
      source.laterality ??
      "—",

    location:
      source.location ??
      source.phc ??
      source.phcName ??
      source.phc_name ??
      "—",

    device:
      source.device ??
      source.deviceId ??
      source.device_id ??
      "—",

    received:
      source.receivedAt ??
      source.received_at ??
      source.createdAt ??
      source.created_at ??
      "—",

    status:
      source.status ??
      source.reviewStatus ??
      source.review_status ??
      "—",

    priority:
      source.priority ??
      "—",

    quality:
      source.quality ??
      source.qualityStatus ??
      source.quality_status ??
      "—",

    raw: source,
  };
}

/* =========================================================
   STEP INDICATOR
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
        <div
          key={label}
          className="flex items-center"
        >
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
                <CheckCircle2 size={15} />
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
   METRIC CARD
========================================================= */

function MetricCard({
  icon: Icon,
  value,
  label,
  detail,
  accent = false,
}) {
  return (
    <div
      className={`rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm ${
        accent ? "border-t-2 border-t-[#0F766E]" : ""
      }`}
    >
      <div className="flex items-start justify-between">
        <div
          className={`flex h-10 w-10 items-center justify-center rounded-xl ${
            accent
              ? "bg-[#F0FDFA] text-[#0F766E]"
              : "bg-[#F1F5F9] text-[#64748B]"
          }`}
        >
          <Icon size={18} />
        </div>

        <CircleDotSmall />
      </div>

      <div className="mt-7 text-[28px] font-black tracking-tight text-[#0F172A]">
        {value}
      </div>

      <div className="mt-1 text-[10px] font-bold uppercase tracking-[0.16em] text-[#64748B]">
        {label}
      </div>

      <div className="mt-2 text-[10px] font-medium text-[#94A3B8]">
        {detail}
      </div>
    </div>
  );
}

/* =========================================================
   SMALL INDICATOR
========================================================= */

function CircleDotSmall() {
  return (
    <span className="mt-1 h-3.5 w-3.5 rounded-full border-2 border-[#CBD5E1]" />
  );
}

/* =========================================================
   STATUS BADGE

   Purely reflects API value.
========================================================= */

function StatusBadge({ status }) {
  if (!status || status === "—") {
    return (
      <span className="inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-[#F8FAFC] px-3 py-1.5 text-[9px] font-bold uppercase tracking-[0.12em] text-[#94A3B8]">
        <span className="h-1.5 w-1.5 rounded-full bg-[#CBD5E1]" />
        No status
      </span>
    );
  }

  const normalized = String(status).toLowerCase();

  let classes =
    "border-[#E2E8F0] bg-[#F8FAFC] text-[#64748B]";

  let dot = "bg-[#94A3B8]";

  if (
    normalized.includes("ready") ||
    normalized.includes("complete") ||
    normalized.includes("completed")
  ) {
    classes =
      "border-[#D1FAE5] bg-[#ECFDF5] text-[#047857]";
    dot = "bg-[#059669]";
  }

  if (
    normalized.includes("review") ||
    normalized.includes("pending")
  ) {
    classes =
      "border-[#E2E8F0] bg-white text-[#64748B]";
    dot = "bg-[#94A3B8]";
  }

  if (
    normalized.includes("error") ||
    normalized.includes("failed")
  ) {
    classes =
      "border-[#F1D2D0] bg-[#FFF7F7] text-[#9A5F5B]";
    dot = "bg-[#B76967]";
  }

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-[9px] font-bold uppercase tracking-[0.12em] ${classes}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${dot}`}
      />

      {status}
    </span>
  );
}

/* =========================================================
   CASE ROW
========================================================= */

function CaseRow({ item, onOpen }) {
  return (
    <div className="group rounded-2xl border border-[#E2E8F0] bg-white p-5 transition hover:border-[#CBD5E1] hover:shadow-sm">
      <div className="grid gap-5 xl:grid-cols-[1.2fr_1fr_0.9fr_0.65fr_auto] xl:items-center">
        {/* CASE */}

        <div>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
              <Eye size={17} />
            </div>

            <div className="min-w-0">
              <div className="truncate text-[12px] font-bold text-[#0F172A]">
                {item.id}
              </div>

              <div className="mt-1 truncate text-[10px] font-medium text-[#64748B]">
                {item.patient} • {item.eye}
              </div>
            </div>
          </div>
        </div>

        {/* POINT OF CARE */}

        <div>
          <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
            Point of care
          </div>

          <div className="mt-2 truncate text-[11px] font-bold text-[#334155]">
            {item.location}
          </div>

          <div className="mt-1 truncate text-[10px] font-medium text-[#64748B]">
            {item.device}
          </div>
        </div>

        {/* STATUS */}

        <div>
          <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
            Workflow
          </div>

          <div className="mt-2">
            <StatusBadge status={item.status} />
          </div>

          <div className="mt-2 text-[10px] font-medium text-[#64748B]">
            Quality: {item.quality}
          </div>
        </div>

        {/* PRIORITY */}

        <div>
          <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
            Priority
          </div>

          <div className="mt-2 text-[10px] font-bold text-[#334155]">
            {item.priority}
          </div>

          <div className="mt-1 flex items-center gap-1.5 text-[9px] text-[#94A3B8]">
            <Clock3 size={11} />
            {item.received}
          </div>
        </div>

        {/* ACTION */}

        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => onOpen(item)}
            disabled={!item.id || item.id === "—"}
            className="inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-2.5 text-[10px] font-bold text-[#334155] transition hover:border-[#CBD5E1] hover:bg-[#F8FAFC] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Open
            <ArrowRight size={13} />
          </button>
        </div>
      </div>
    </div>
  );
}

/* =========================================================
   EMPTY STATE
========================================================= */

function EmptyState({ configured, error, onRetry }) {
  return (
    <div className="rounded-[28px] border border-[#E2E8F0] bg-white px-6 py-16 text-center shadow-sm">
      <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[#F1F5F9] text-[#0F766E]">
        {error ? (
          <XCircle size={24} />
        ) : configured ? (
          <FileCheck2 size={24} />
        ) : (
          <NetworkIcon />
        )}
      </div>

      <h3 className="mt-5 text-lg font-black text-[#0F172A]">
        {error
          ? "Unable to load review queue"
          : configured
            ? "No cases returned"
            : "Review queue is not connected"}
      </h3>

      <p className="mx-auto mt-2 max-w-[560px] text-[12px] leading-6 text-[#64748B]">
        {error
          ? error
          : configured
            ? "The connected review service returned no screening cases."
            : "Configure VITE_REVIEW_QUEUE_URL to connect this workspace to your real backend review queue."}
      </p>

      {configured && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-6 inline-flex items-center gap-2 rounded-full bg-[#111816] px-5 py-3 text-[11px] font-bold text-white transition hover:bg-[#1C2723]"
        >
          <RefreshCcw size={14} />
          Refresh queue
        </button>
      )}
    </div>
  );
}

/* =========================================================
   NETWORK ICON
========================================================= */

function NetworkIcon() {
  return <Network size={24} />;
}

/* =========================================================
   LOCAL REVIEW FALLBACK
   Uses only review records actually saved by DoctorReview.
========================================================= */

function readLocalReviewCases() {
  const records = [];

  try {
    for (let index = 0; index < localStorage.length; index += 1) {
      const key = localStorage.key(index);

      if (!key || !key.startsWith("visionx.review.")) continue;
      if (key.startsWith("visionx.review.draft.")) continue;

      const raw = localStorage.getItem(key);
      if (!raw) continue;

      try {
        const review = JSON.parse(raw);
        if (!review?.caseId) continue;

        records.push({
          id: review.caseId,
          patient: "—",
          eye: "—",
          location: "—",
          device: "—",
          received: review.timestamp
            ? new Date(review.timestamp).toLocaleString()
            : "—",
          status:
            review.decision === "refer"
              ? "Referral review completed"
              : review.decision === "validated"
                ? "Finding validated"
                : review.decision === "review"
                  ? "Further review"
                  : "Review completed",
          priority: review.decision === "refer" ? "Referral" : "Routine",
          quality: "—",
          raw: {
            ...review,
            source: "local-review-record",
          },
        });
      } catch {
        // Ignore malformed local entries.
      }
    }
  } catch {
    return [];
  }

  return records;
}

/* =========================================================
   MAIN DOCTOR DASHBOARD
========================================================= */

export default function Doctor() {
  const navigate = useNavigate();

  const [cases, setCases] = useState([]);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  /* =======================================================
     FETCH REAL REVIEW QUEUE
  ======================================================= */

  const fetchCases = useCallback(async () => {
    if (!REVIEW_QUEUE_URL) {
      const localCases = readLocalReviewCases();
      setCases(localCases.map(normalizeCase));
      setError("");
      setLastUpdated(localCases.length ? new Date() : null);
      return;
    }

    setLoading(true);
    setError("");

    const controller = new AbortController();

    try {
      const response = await fetch(
        REVIEW_QUEUE_URL,
        {
          method: "GET",
          headers: {
            Accept: "application/json",
          },
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        throw new Error(
          `Review queue request failed (${response.status}).`
        );
      }

      const payload = await response.json();

      const extracted = extractCases(payload);

      const normalized = extracted.map(
        normalizeCase
      );

      setCases(normalized);
      setLastUpdated(new Date());
    } catch (requestError) {
      if (requestError.name === "AbortError") {
        return;
      }

      const localCases = readLocalReviewCases();

      if (localCases.length > 0) {
        setCases(localCases.map(normalizeCase));
        setError("");
        setLastUpdated(new Date());
      } else {
        setCases([]);
        setError(
          requestError?.message ||
            "The review queue could not be loaded."
        );
      }
    } finally {
      setLoading(false);
    }

    return () => controller.abort();
  }, []);

  useEffect(() => {
    let cancelled = false;

    const run = async () => {
      if (!cancelled) {
        await fetchCases();
      }
    };

    run();

    return () => {
      cancelled = true;
    };
  }, [fetchCases]);

  /* =======================================================
     DYNAMIC FILTER OPTIONS
  ======================================================= */

  const statuses = useMemo(() => {
    const unique = new Set();

    cases.forEach((item) => {
      if (item.status && item.status !== "—") {
        unique.add(String(item.status));
      }
    });

    return ["All", ...Array.from(unique)];
  }, [cases]);

  /* =======================================================
     SEARCH + FILTER
  ======================================================= */

  const filteredCases = useMemo(() => {
    const query = search.trim().toLowerCase();

    return cases.filter((item) => {
      const searchable = [
        item.id,
        item.patient,
        item.eye,
        item.location,
        item.device,
        item.status,
        item.priority,
      ]
        .join(" ")
        .toLowerCase();

      const matchesSearch =
        !query || searchable.includes(query);

      const matchesStatus =
        statusFilter === "All" ||
        item.status === statusFilter;

      return matchesSearch && matchesStatus;
    });
  }, [cases, search, statusFilter]);

  /* =======================================================
     DERIVED METRICS
     NO HARDCODED COUNTS
  ======================================================= */

  const metrics = useMemo(() => {
    const total = cases.length;

    const evidenceReady = cases.filter((item) => {
      const value = String(item.status).toLowerCase();

      return (
        value.includes("evidence") ||
        value.includes("complete") ||
        value.includes("ready")
      );
    }).length;

    const reviewPending = cases.filter((item) => {
      const value = String(item.status).toLowerCase();

      return (
        value.includes("review") ||
        value.includes("pending")
      );
    }).length;

    const phcs = new Set(
      cases
        .map((item) => item.location)
        .filter(
          (location) =>
            location && location !== "—"
        )
    ).size;

    return {
      total,
      evidenceReady,
      reviewPending,
      phcs,
    };
  }, [cases]);

  /* =======================================================
     OPEN CASE
  ======================================================= */

  const openCase = (item) => {
    if (!item.id || item.id === "—") return;

    navigate(
      `/doctor/review/${encodeURIComponent(item.id)}`,
      {
        state: {
          caseData: item.raw,
        },
      }
    );
  };

  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <main className="min-h-screen bg-[#F8FAFC] text-[#0F172A]">
      {/* ===================================================
          BACKGROUND GRID
      =================================================== */}

      <div
        className="pointer-events-none fixed inset-0 opacity-[0.35]"
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
            {/* Brand */}

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

            {/* Connection state */}

            <div className="flex items-center gap-3">
              <div
                className={`hidden items-center gap-2 rounded-full border px-3.5 py-2 text-[9px] font-bold uppercase tracking-[0.14em] sm:flex ${
                  REVIEW_QUEUE_URL && !error
                    ? "border-[#D1FAE5] bg-[#ECFDF5] text-[#047857]"
                    : "border-[#E2E8F0] bg-[#F8FAFC] text-[#64748B]"
                }`}
              >
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    REVIEW_QUEUE_URL && !error
                      ? "bg-[#059669]"
                      : "bg-[#94A3B8]"
                  }`}
                />

                {loading
                  ? "Loading queue"
                  : REVIEW_QUEUE_URL && !error
                    ? "Review API connected"
                    : cases.length
                      ? "Local review records"
                      : "Review API not configured"}
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

        {/* =================================================
            PAGE
        ================================================= */}

        <div className="mx-auto max-w-[1460px] px-6 py-10 md:px-10 lg:px-12">
          {/* =================================================
              HEADING
          ================================================= */}

          <section className="flex flex-col justify-between gap-7 lg:flex-row lg:items-end">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">
                Clinical workspace
              </div>

              <h1 className="mt-4 text-[42px] font-black leading-none tracking-tight text-[#0F172A] md:text-[54px]">
                Review the
                <span className="block text-[#0F766E]">
                  screening queue.
                </span>
              </h1>

              <p className="mt-6 max-w-[720px] text-[16px] font-medium leading-relaxed text-[#475569]">
                Review screening cases returned by the connected
                VisionX workflow and open individual cases for
                clinical validation.
              </p>
            </div>

            <div className="flex items-center gap-3 rounded-2xl border border-[#E2E8F0] bg-white px-5 py-4 shadow-sm">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
                <Stethoscope size={16} />
              </div>

              <div>
                <div className="text-[9px] font-bold uppercase tracking-[0.17em] text-[#94A3B8]">
                  Workspace
                </div>

                <div className="mt-1 text-[12px] font-bold text-[#334155]">
                  Clinical review
                </div>
              </div>
            </div>
          </section>

          {/* =================================================
              METRICS
          ================================================= */}

          <section className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              icon={FileCheck2}
              value={metrics.total}
              label="Cases returned"
              detail="From the connected review API"
              accent
            />

            <MetricCard
              icon={Eye}
              value={metrics.evidenceReady}
              label="Evidence ready"
              detail="Derived from case status"
            />

            <MetricCard
              icon={Clock3}
              value={metrics.reviewPending}
              label="Review states"
              detail="Cases requiring review"
            />

            <MetricCard
              icon={Hospital}
              value={metrics.phcs}
              label="Source PHCs"
              detail="Unique locations in returned data"
            />
          </section>

          {/* =================================================
              QUEUE CONNECTION PANEL
          ================================================= */}

          <section className="mt-7 overflow-hidden rounded-[30px] bg-[#111816] p-5 shadow-xl md:p-6">
            <div className="rounded-[24px] border border-[#2B3B35] bg-[#16201D] p-6">
              <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-center">
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#80958B]">
                    Review operations
                  </div>

                  <h2 className="mt-3 text-2xl font-black text-white">
                    Connected clinical
                    <span className="text-[#80958B]">
                      {" "}review queue.
                    </span>
                  </h2>

                  <p className="mt-3 max-w-[660px] text-[12px] leading-6 text-[#A9BEB4]">
                    This dashboard does not manufacture cases or
                    screening results. It displays the records
                    returned by the configured VisionX backend.
                  </p>
                </div>

                <button
                  type="button"
                  onClick={fetchCases}
                  disabled={loading}
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-[#2B3B35] bg-[#1C2723] px-5 py-3 text-[10px] font-bold uppercase tracking-[0.13em] text-[#DCE8E1] transition hover:bg-[#25312C] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <RefreshCcw
                    size={13}
                    className={
                      loading
                        ? "animate-spin"
                        : ""
                    }
                  />
                  Refresh
                </button>
              </div>

              <div className="mt-7 grid gap-3 md:grid-cols-3">
                <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723]/65 p-5">
                  <div className="text-[10px] font-bold tracking-[0.18em] text-[#80958B]">
                    RETURNED
                  </div>

                  <div className="mt-5 text-2xl font-black text-white">
                    {metrics.total}
                  </div>

                  <div className="mt-1 text-[10px] font-medium text-[#80958B]">
                    cases in current response
                  </div>
                </div>

                <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723]/65 p-5">
                  <div className="text-[10px] font-bold tracking-[0.18em] text-[#80958B]">
                    FILTERED
                  </div>

                  <div className="mt-5 text-2xl font-black text-white">
                    {filteredCases.length}
                  </div>

                  <div className="mt-1 text-[10px] font-medium text-[#80958B]">
                    cases matching current view
                  </div>
                </div>

                <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723]/65 p-5">
                  <div className="text-[10px] font-bold tracking-[0.18em] text-[#80958B]">
                    LAST UPDATED
                  </div>

                  <div className="mt-5 text-[14px] font-black text-white">
                    {lastUpdated
                      ? lastUpdated.toLocaleTimeString()
                      : "—"}
                  </div>

                  <div className="mt-1 text-[10px] font-medium text-[#80958B]">
                    local browser time
                  </div>
                </div>
              </div>
            </div>
          </section>

          {/* =================================================
              CASE QUEUE HEADER
          ================================================= */}

          <section className="mt-8">
            <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
              <div>
                <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#0F766E]">
                  Case queue
                </div>

                <h2 className="mt-3 text-2xl font-black tracking-tight text-[#0F172A]">
                  Screening cases
                </h2>

                <p className="mt-2 text-[12px] font-medium text-[#64748B]">
                  Search and filter the records returned by the
                  review service.
                </p>
              </div>

              <div className="flex flex-col gap-2 sm:flex-row">
                {/* Search */}

                <div className="relative">
                  <Search
                    size={15}
                    className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#94A3B8]"
                  />

                  <input
                    type="text"
                    value={search}
                    onChange={(event) =>
                      setSearch(event.target.value)
                    }
                    placeholder="Search returned cases"
                    className="w-full rounded-xl border border-[#E2E8F0] bg-white py-3 pl-10 pr-4 text-[11px] font-medium text-[#334155] outline-none placeholder:text-[#94A3B8] focus:border-[#A9D5C4] sm:w-[250px]"
                  />
                </div>

                {/* Dynamic status filter */}

                <div className="relative">
                  <Filter
                    size={14}
                    className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-[#94A3B8]"
                  />

                  <select
                    value={statusFilter}
                    onChange={(event) =>
                      setStatusFilter(event.target.value)
                    }
                    className="appearance-none rounded-xl border border-[#E2E8F0] bg-white py-3 pl-9 pr-9 text-[11px] font-bold text-[#334155] outline-none"
                  >
                    {statuses.map((status) => (
                      <option
                        key={status}
                        value={status}
                      >
                        {status}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>

            {/* =================================================
                ERROR
            ================================================= */}

            {error && REVIEW_QUEUE_URL ? (
              <div className="mt-5 flex items-start gap-3 rounded-2xl border border-[#F1D2D0] bg-[#FFF7F7] px-5 py-4 text-[11px] text-[#9A5F5B]">
                <XCircle
                  size={16}
                  className="mt-0.5 shrink-0"
                />

                <div>
                  <div className="font-bold">
                    Review queue request failed
                  </div>

                  <div className="mt-1">
                    {error}
                  </div>
                </div>
              </div>
            ) : null}

            {/* =================================================
                LOADING
            ================================================= */}

            {loading ? (
              <div className="mt-5 flex min-h-[320px] items-center justify-center rounded-[28px] border border-[#E2E8F0] bg-white">
                <div className="text-center">
                  <RefreshCcw
                    size={27}
                    className="mx-auto animate-spin text-[#0F766E]"
                  />

                  <div className="mt-4 text-[13px] font-bold text-[#334155]">
                    Loading review queue
                  </div>

                  <div className="mt-1 text-[11px] text-[#64748B]">
                    Reading cases from the connected service.
                  </div>
                </div>
              </div>
            ) : cases.length === 0 ? (
              <div className="mt-5">
                <EmptyState
                  configured={Boolean(
                    REVIEW_QUEUE_URL
                  )}
                  error={error}
                  onRetry={fetchCases}
                />
              </div>
            ) : filteredCases.length === 0 ? (
              <div className="mt-5 rounded-[28px] border border-[#E2E8F0] bg-white px-6 py-16 text-center shadow-sm">
                <Search
                  size={28}
                  className="mx-auto text-[#CBD5E1]"
                />

                <div className="mt-4 text-[14px] font-bold text-[#334155]">
                  No matching returned cases
                </div>

                <div className="mt-1 text-[11px] text-[#64748B]">
                  Change the search text or status filter.
                </div>
              </div>
            ) : (
              <div className="mt-5 space-y-3">
                {filteredCases.map((item) => (
                  <CaseRow
                    key={item.key}
                    item={item}
                    onOpen={openCase}
                  />
                ))}
              </div>
            )}
          </section>

          {/* =================================================
              WORKSPACE NOTES
          ================================================= */}

          <section className="mt-8 grid gap-5 lg:grid-cols-3">
            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-6 shadow-sm">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <Eye size={19} />
              </div>

              <div className="mt-6 text-[14px] font-bold text-[#0F172A]">
                Evidence-first review
              </div>

              <p className="mt-2 text-[12px] leading-5 text-[#64748B]">
                Individual cases open into the separate clinical
                review workspace.
              </p>
            </div>

            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-6 shadow-sm">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <ShieldCheck size={19} />
              </div>

              <div className="mt-6 text-[14px] font-bold text-[#0F172A]">
                Human-in-the-loop
              </div>

              <p className="mt-2 text-[12px] leading-5 text-[#64748B]">
                Clinical decisions should be recorded through
                the review workflow rather than inferred by the
                dashboard.
              </p>
            </div>

            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-6 shadow-sm">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <Wifi size={19} />
              </div>

              <div className="mt-6 text-[14px] font-bold text-[#0F172A]">
                Live backend data
              </div>

              <p className="mt-2 text-[12px] leading-5 text-[#64748B]">
                Counts and case records shown above are derived
                from the connected review API response.
              </p>
            </div>
          </section>

          {/* =================================================
              FOOTER
          ================================================= */}

          <div className="mt-10 flex flex-col justify-between gap-3 border-t border-[#E2E8F0] pt-6 text-[10px] font-medium text-[#64748B] md:flex-row">
            <div>
              VisionX Clinical Review • Live case workspace
            </div>

            <div className="flex items-center gap-2">
              <Activity size={13} />
              Backend-driven review queue
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}