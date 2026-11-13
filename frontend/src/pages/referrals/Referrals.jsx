import { useCallback, useEffect, useMemo, useState } from "react";

import {
  ArrowRight,
  CheckCircle2,
  FileCheck2,
  RefreshCcw,
  Search,
} from "lucide-react";

import PageShell from "../../components/layout/PageShell";

const REFERRALS_URL = import.meta.env.VITE_REFERRALS_URL || "";

function extractReferrals(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.referrals)) return payload.referrals;
  if (Array.isArray(payload?.data)) return payload.data;
  if (Array.isArray(payload?.results)) return payload.results;
  return [];
}

function readLocalReviewReferrals() {
  const records = [];

  try {
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);

      if (
        !key ||
        !key.startsWith("visionx.review.") ||
        key.startsWith("visionx.review.draft.")
      ) {
        continue;
      }

      const raw = localStorage.getItem(key);
      if (!raw) continue;

      try {
        const review = JSON.parse(raw);

        if (review?.caseId && review?.decision === "refer") {
          records.push({
            id: `REF-${review.caseId}`,
            caseId: review.caseId,
            patientName: "—",
            status: "Referral pending",
            reviewer: review.reviewer ?? "Clinical workspace",
            reviewerNote: review.reviewerNote ?? "",
            createdAt: review.timestamp ?? null,
            source: "local-review-record",
          });
        }
      } catch {
        // Ignore malformed records.
      }
    }
  } catch {
    // localStorage may be unavailable.
  }

  return records.sort((a, b) => {
    const at = a.createdAt ? Date.parse(a.createdAt) : 0;
    const bt = b.createdAt ? Date.parse(b.createdAt) : 0;
    return bt - at;
  });
}

function normalizeReferral(item, index) {
  const source = item || {};

  return {
    key:
      source.id ??
      source.referralId ??
      source.referral_id ??
      `record-${index}`,
    id:
      source.id ??
      source.referralId ??
      source.referral_id ??
      "—",
    caseId:
      source.caseId ??
      source.case_id ??
      "—",
    patient:
      source.patientName ??
      source.patient_name ??
      source.patientId ??
      source.patient_id ??
      "—",
    status: source.status ?? "—",
    createdAt:
      source.createdAt ??
      source.created_at ??
      source.timestamp ??
      null,
    reviewer: source.reviewer ?? "—",
    note: source.reviewerNote ?? source.note ?? "",
    raw: source,
  };
}

export default function Referrals() {
  const [referrals, setReferrals] = useState([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [source, setSource] = useState("none");

  const fetchReferrals = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      if (REFERRALS_URL) {
        const response = await fetch(REFERRALS_URL, {
          headers: { Accept: "application/json" },
        });

        if (!response.ok) {
          throw new Error(`Referral request failed (${response.status}).`);
        }

        setReferrals(extractReferrals(await response.json()));
        setSource("api");
        return;
      }

      setReferrals(readLocalReviewReferrals());
      setSource("local");
    } catch (err) {
      const local = readLocalReviewReferrals();

      if (local.length > 0) {
        setReferrals(local);
        setSource("local");
        setError("");
      } else {
        setReferrals([]);
        setSource("none");
        setError(err?.message || "Unable to load referrals.");
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchReferrals();

    const onStorage = () => {
      if (!REFERRALS_URL) {
        setReferrals(readLocalReviewReferrals());
        setSource("local");
      }
    };

    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [fetchReferrals]);

  const normalized = useMemo(
    () => referrals.map(normalizeReferral),
    [referrals]
  );

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) return normalized;

    return normalized.filter((item) =>
      JSON.stringify(item.raw).toLowerCase().includes(query) ||
      item.id.toLowerCase().includes(query) ||
      item.caseId.toLowerCase().includes(query)
    );
  }, [normalized, search]);

  return (
    <PageShell
      eyebrow="Connected care"
      title="Track the"
      accentTitle="referral pathway."
      description="Track referral records returned by the care service or created by a completed local review during this demo session."
      rightContent={
        <button
          type="button"
          onClick={fetchReferrals}
          disabled={loading}
          className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#64748B] disabled:opacity-40"
        >
          <RefreshCcw
            size={15}
            className={loading ? "animate-spin" : ""}
          />
        </button>
      }
    >
      <section className="mt-8 flex flex-col justify-between gap-4 md:flex-row md:items-center">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#0F766E]">
            Referral records
          </div>

          <h2 className="mt-2 text-2xl font-black text-[#0F172A]">
            Care transitions
          </h2>

          {source === "local" && (
            <div className="mt-2 inline-flex items-center gap-2 rounded-full border border-[#D1FAE5] bg-[#ECFDF5] px-3 py-1.5 text-[9px] font-bold uppercase tracking-[0.12em] text-[#047857]">
              <CheckCircle2 size={11} />
              Demo-session referrals
            </div>
          )}
        </div>

        <div className="relative">
          <Search
            size={15}
            className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#94A3B8]"
          />

          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search referrals"
            className="rounded-xl border border-[#E2E8F0] bg-white py-3 pl-10 pr-4 text-[11px] font-medium text-[#334155] outline-none placeholder:text-[#94A3B8] focus:border-[#A9D5C4] sm:w-[250px]"
          />
        </div>
      </section>

      {error ? (
        <div className="mt-5 rounded-2xl border border-[#F1D2D0] bg-[#FFF7F7] p-4 text-[11px] text-[#9A5F5B]">
          {error}
        </div>
      ) : null}

      {loading ? (
        <div className="mt-5 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
          <RefreshCcw
            size={28}
            className="mx-auto animate-spin text-[#0F766E]"
          />

          <div className="mt-4 text-[13px] font-bold text-[#334155]">
            Loading referral records
          </div>
        </div>
      ) : filtered.length === 0 ? (
        <div className="mt-5 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
          <FileCheck2
            size={30}
            className="mx-auto text-[#CBD5E1]"
          />

          <div className="mt-4 text-[14px] font-bold text-[#334155]">
            No referral records available
          </div>

          <div className="mx-auto mt-2 max-w-[540px] text-[11px] leading-5 text-[#64748B]">
            A referral appears here when a clinical review is completed with
            the “Refer case” disposition, or when a real referral API is
            connected.
          </div>
        </div>
      ) : (
        <div className="mt-5 space-y-3">
          {filtered.map((item) => (
            <div
              key={item.key}
              className="rounded-2xl border border-[#E2E8F0] bg-white p-5"
            >
              <div className="grid gap-5 md:grid-cols-[1fr_1fr_0.8fr_1fr_auto] md:items-center">
                <div>
                  <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                    Referral ID
                  </div>

                  <div className="mt-2 text-[12px] font-bold text-[#0F172A]">
                    {item.id}
                  </div>
                </div>

                <div>
                  <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                    Case
                  </div>

                  <div className="mt-2 text-[11px] font-bold text-[#334155]">
                    {item.caseId}
                  </div>
                </div>

                <div>
                  <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                    Patient
                  </div>

                  <div className="mt-2 text-[11px] font-bold text-[#334155]">
                    {item.patient}
                  </div>
                </div>

                <div>
                  <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                    Status
                  </div>

                  <div className="mt-2 text-[11px] font-bold text-[#334155]">
                    {item.status}
                  </div>

                  {item.createdAt && (
                    <div className="mt-1 text-[9px] text-[#94A3B8]">
                      {new Date(item.createdAt).toLocaleString()}
                    </div>
                  )}
                </div>

                <button
                  type="button"
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-[#E2E8F0] px-4 py-2.5 text-[10px] font-bold text-[#334155]"
                  title="Referral details are read-only in this demo session."
                >
                  Details
                  <ArrowRight size={13} />
                </button>
              </div>

              {item.note && (
                <div className="mt-4 rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] px-4 py-3">
                  <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
                    Reviewer note
                  </div>
                  <div className="mt-1 text-[10px] leading-5 text-[#475569]">
                    {item.note}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      <div className="mt-8 rounded-2xl border border-[#E2E8F0] bg-white p-5 text-[11px] text-[#64748B]">
        <FileCheck2
          size={14}
          className="mr-2 inline text-[#0F766E]"
        />
        {source === "api"
          ? "Referral values above are derived from the connected referral service."
          : "With no referral backend configured, this workspace reads only completed “Refer case” review records saved in this browser."}
      </div>
    </PageShell>
  );
}
