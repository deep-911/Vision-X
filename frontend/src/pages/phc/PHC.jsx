import { useCallback, useEffect, useState } from "react";
import { Activity, Hospital, RefreshCcw, Wifi, Server, ShieldCheck } from "lucide-react";
import PageShell from "../../components/layout/PageShell";

const HEALTH_URL =
  import.meta.env.VITE_HEALTH_URL || "http://127.0.0.1:8000/api/health";

function readLocalReviews() {
  const records = [];

  try {
    for (let i = 0; i < localStorage.length; i += 1) {
      const key = localStorage.key(i);
      if (!key || !key.startsWith("visionx.review.") || key.startsWith("visionx.review.draft.")) continue;

      const raw = localStorage.getItem(key);
      if (!raw) continue;

      try {
        const review = JSON.parse(raw);
        if (review?.caseId) records.push(review);
      } catch {
        // Ignore malformed local records.
      }
    }
  } catch {
    // localStorage may be unavailable.
  }

  return records;
}

export default function PHC() {
  const [health, setHealth] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchStatus = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(HEALTH_URL, {
        headers: { Accept: "application/json" },
      });

      if (!response.ok) {
        throw new Error(`VisionX service request failed (${response.status}).`);
      }

      setHealth(await response.json());
      setReviews(readLocalReviews());
    } catch (err) {
      setHealth(null);
      setReviews(readLocalReviews());
      setError(err?.message || "Unable to load VisionX service status.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchStatus();

    const onStorage = () => setReviews(readLocalReviews());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, [fetchStatus]);

  const backendReady = health?.status === "ok" && health?.readyForInference === true;

  return (
    <PageShell
      eyebrow="Distributed care"
      title="Operate the"
      accentTitle="PHC network."
      description="Monitor the connected VisionX inference service and local screening workflow. Live PHC/device telemetry is shown only when a real operations service is connected."
      rightContent={
        <button
          type="button"
          onClick={fetchStatus}
          disabled={loading}
          className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#64748B] disabled:opacity-40"
        >
          <RefreshCcw size={15} className={loading ? "animate-spin" : ""} />
        </button>
      }
    >
      {loading ? (
        <div className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
          <RefreshCcw size={28} className="mx-auto animate-spin text-[#0F766E]" />
          <div className="mt-4 text-[13px] font-bold text-[#334155]">
            Checking VisionX service
          </div>
        </div>
      ) : (
        <>
          {error && (
            <div className="mt-8 rounded-2xl border border-[#F1D2D0] bg-[#FFF7F7] p-5 text-[11px] text-[#9A5F5B]">
              {error}
            </div>
          )}

          <section className="mt-8 grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm">
              <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                Inference service
              </div>
              <div className="mt-3 flex items-center gap-2 text-[20px] font-black text-[#0F172A]">
                <span className={`h-2.5 w-2.5 rounded-full ${health ? "bg-[#059669]" : "bg-[#B76967]"}`} />
                {health?.inference ?? "Unavailable"}
              </div>
            </div>

            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm">
              <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                Model readiness
              </div>
              <div className="mt-3 text-[27px] font-black text-[#0F172A]">
                {backendReady ? "READY" : health ? "CHECK" : "—"}
              </div>
            </div>

            <div className="rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm">
              <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                Local reviewed cases
              </div>
              <div className="mt-3 text-[27px] font-black text-[#0F172A]">
                {reviews.length}
              </div>
            </div>
          </section>

          <section className="mt-6 grid gap-5 lg:grid-cols-2">
            <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
                  <Server size={19} />
                </div>
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#64748B]">
                    Connected backend
                  </div>
                  <div className="mt-1 text-[16px] font-black text-[#0F172A]">
                    VisionX inference node
                  </div>
                </div>
              </div>

              <div className="mt-6 space-y-2.5">
                <div className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3">
                  <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94A3B8]">
                    PyTorch
                  </span>
                  <span className="text-[11px] font-bold text-[#334155]">
                    {health?.torchInstalled ? "Available" : "Unavailable"}
                  </span>
                </div>
                <div className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3">
                  <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94A3B8]">
                    DR model
                  </span>
                  <span className="text-[11px] font-bold text-[#334155]">
                    {health?.drModelExists ? "Loaded" : "Unavailable"}
                  </span>
                </div>
                <div className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3">
                  <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#94A3B8]">
                    Lesion model
                  </span>
                  <span className="text-[11px] font-bold text-[#334155]">
                    {health?.lesionModelExists ? "Loaded" : "Unavailable"}
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
                  <Hospital size={19} />
                </div>
                <div>
                  <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#64748B]">
                    Point-of-care telemetry
                  </div>
                  <div className="mt-1 text-[16px] font-black text-[#0F172A]">
                    Live PHC service not connected
                  </div>
                </div>
              </div>

              <div className="mt-6 rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] p-5">
                <div className="flex items-start gap-3">
                  <Wifi size={17} className="mt-0.5 text-[#0F766E]" />
                  <p className="text-[11px] leading-5 text-[#64748B]">
                    No PHC location, device, workload, or connectivity values are invented here.
                    Connect a real PHC operations endpoint when available.
                  </p>
                </div>
              </div>

              <div className="mt-4 flex items-start gap-3 rounded-2xl border border-[#D1FAE5] bg-[#ECFDF5] p-5">
                <ShieldCheck size={17} className="mt-0.5 text-[#059669]" />
                <p className="text-[11px] leading-5 text-[#047857]">
                  The screening and review workflow remains functional independently of PHC telemetry.
                </p>
              </div>
            </div>
          </section>

          {health && (
            <div className="mt-8 rounded-2xl border border-[#E2E8F0] bg-white p-5 text-[11px] text-[#64748B]">
              <Activity size={14} className="mr-2 inline text-[#0F766E]" />
              VisionX backend status is read directly from the connected local inference service.
            </div>
          )}
        </>
      )}
    </PageShell>
  );
}
