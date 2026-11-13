import { useCallback, useEffect, useState } from "react";

import {
  CheckCircle2,
  Database,
  RefreshCcw,
  ShieldCheck,
} from "lucide-react";

import PageShell from "../../components/layout/PageShell";

const VALIDATION_URL =
  import.meta.env.VITE_VALIDATION_URL || "";

export default function Validation() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchValidation = useCallback(async () => {
    if (!VALIDATION_URL) {
      setData(null);
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(VALIDATION_URL, {
        headers: {
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(
          `Validation request failed (${response.status}).`
        );
      }

      setData(await response.json());
    } catch (err) {
      setData(null);
      setError(
        err?.message || "Unable to load validation data."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchValidation();
  }, [fetchValidation]);

  return (
    <PageShell
      eyebrow="Model validation"
      title="Inspect the"
      accentTitle="evidence behind performance."
      description="Review validation metadata and model evaluation results returned by the connected VisionX validation service."
      rightContent={
        <button
          type="button"
          onClick={fetchValidation}
          disabled={!VALIDATION_URL || loading}
          className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#64748B] disabled:opacity-40"
        >
          <RefreshCcw
            size={15}
            className={loading ? "animate-spin" : ""}
          />
        </button>
      }
    >
      {!VALIDATION_URL ? (
        <div className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
          <Database
            size={30}
            className="mx-auto text-[#CBD5E1]"
          />

          <div className="mt-4 text-[14px] font-bold text-[#334155]">
            Validation service not connected
          </div>

          <div className="mt-2 text-[11px] text-[#64748B]">
            Configure VITE_VALIDATION_URL to display actual
            validation records.
          </div>
        </div>
      ) : loading ? (
        <div className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
          <RefreshCcw
            size={28}
            className="mx-auto animate-spin text-[#0F766E]"
          />

          <div className="mt-4 text-[13px] font-bold text-[#334155]">
            Loading validation results
          </div>
        </div>
      ) : error ? (
        <div className="mt-8 rounded-2xl border border-[#F1D2D0] bg-[#FFF7F7] p-5 text-[11px] text-[#9A5F5B]">
          {error}
        </div>
      ) : !data ? (
        <div className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center text-[11px] text-[#64748B]">
          No validation data returned.
        </div>
      ) : (
        <>
          <section className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
            {[
              ["Dataset", data?.dataset],
              ["Model", data?.model],
              ["Version", data?.version],
              ["Evaluation set", data?.evaluationSet ?? data?.evaluation_set],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-2xl border border-[#E2E8F0] bg-white p-5"
              >
                <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                  {label}
                </div>

                <div className="mt-3 text-[14px] font-black text-[#0F172A]">
                  {value ?? "—"}
                </div>
              </div>
            ))}
          </section>

          <section className="mt-6 grid gap-6 lg:grid-cols-2">
            <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <ShieldCheck size={19} />
              </div>

              <h2 className="mt-5 text-xl font-black text-[#0F172A]">
                Evaluation metrics
              </h2>

              <div className="mt-5 space-y-3">
                {Object.entries(
                  data?.metrics || {}
                ).map(([label, value]) => (
                  <div
                    key={label}
                    className="flex items-center justify-between rounded-xl bg-[#F8FAFC] px-4 py-3.5"
                  >
                    <span className="text-[10px] font-bold uppercase tracking-[0.12em] text-[#64748B]">
                      {label}
                    </span>

                    <span className="text-[12px] font-black text-[#0F172A]">
                      {String(value)}
                    </span>
                  </div>
                ))}

                {Object.keys(data?.metrics || {}).length === 0 ? (
                  <div className="rounded-xl bg-[#F8FAFC] p-5 text-[11px] text-[#64748B]">
                    No metric values returned.
                  </div>
                ) : null}
              </div>
            </div>

            <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <Database size={19} />
              </div>

              <h2 className="mt-5 text-xl font-black text-[#0F172A]">
                Validation metadata
              </h2>

              <pre className="mt-5 max-h-[400px] overflow-auto rounded-2xl bg-[#F8FAFC] p-5 text-[10px] leading-5 text-[#64748B]">
                {JSON.stringify(
                  data?.metadata || {},
                  null,
                  2
                )}
              </pre>
            </div>
          </section>

          <div className="mt-6 flex items-start gap-3 rounded-2xl border border-[#E2E8F0] bg-white p-5 text-[11px] text-[#64748B]">
            <CheckCircle2
              size={15}
              className="mt-0.5 shrink-0 text-[#059669]"
            />

            Values displayed on this page are taken from the
            connected validation response.
          </div>
        </>
      )}
    </PageShell>
  );
}