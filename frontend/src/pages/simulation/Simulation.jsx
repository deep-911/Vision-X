import { useState } from "react";

import {
  ArrowRight,
  CheckCircle2,
  Network,
  Play,
  RefreshCcw,
} from "lucide-react";

import PageShell from "../../components/layout/PageShell";

const SIMULATION_URL = import.meta.env.VITE_SIMULATION_URL || "";

function localScenarioCheck(form) {
  const patientsPerDay = Number(form.patientsPerDay);
  const acquisitionRate = Number(form.acquisitionRate);
  const bandwidthMbps = Number(form.bandwidthMbps);
  const reviewCapacity = Number(form.reviewCapacity);

  if (
    !Number.isFinite(patientsPerDay) || patientsPerDay <= 0 ||
    !Number.isFinite(acquisitionRate) || acquisitionRate <= 0 ||
    !Number.isFinite(bandwidthMbps) || bandwidthMbps <= 0 ||
    !Number.isFinite(reviewCapacity) || reviewCapacity <= 0
  ) {
    throw new Error("Enter positive values for all four simulation inputs.");
  }

  const utilization = (patientsPerDay / reviewCapacity) * 100;
  const capacityGap = reviewCapacity - patientsPerDay;

  return {
    simulationType: "LOCAL_SCENARIO_CHECK",
    note: "This is a transparent browser-side capacity check, not a Simulink result.",
    inputs: { patientsPerDay, acquisitionRate, bandwidthMbps, reviewCapacity },
    derived: {
      reviewCapacityUtilizationPercent: Number(utilization.toFixed(1)),
      reviewCapacityGapPerDay: capacityGap,
      capacityStatus:
        capacityGap >= 0 ? "WITHIN_REVIEW_CAPACITY" : "REVIEW_CAPACITY_EXCEEDED",
    },
  };
}

export default function Simulation() {
  const [form, setForm] = useState({
    patientsPerDay: "",
    acquisitionRate: "",
    bandwidthMbps: "",
    reviewCapacity: "",
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateField = (field, value) => {
    setForm((previous) => ({ ...previous, [field]: value }));
  };

  const runSimulation = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      if (SIMULATION_URL) {
        const response = await fetch(SIMULATION_URL, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "application/json",
          },
          body: JSON.stringify(form),
        });

        if (!response.ok) {
          throw new Error(`Simulation request failed (${response.status}).`);
        }

        setResult(await response.json());
      } else {
        setResult(localScenarioCheck(form));
      }
    } catch (err) {
      setError(err?.message || "Unable to run simulation.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageShell
      eyebrow="District modelling"
      title="Model the"
      accentTitle="screening workflow."
      description="Run a connected Simulink simulation when available. Otherwise, VisionX provides a transparent local capacity check from the values you enter."
      rightContent={
        <div className="flex items-center gap-3 rounded-2xl border border-[#E2E8F0] bg-white px-5 py-4 shadow-sm">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
            <Network size={16} />
          </div>
          <div>
            <div className="text-[9px] font-bold uppercase tracking-[0.17em] text-[#94A3B8]">
              Simulation
            </div>
            <div className="mt-1 text-[12px] font-bold text-[#334155]">
              {SIMULATION_URL ? "API driven" : "Local scenario check"}
            </div>
          </div>
        </div>
      }
    >
      <section className="mt-8 grid gap-6 lg:grid-cols-[0.8fr_1.2fr]">
        <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
          <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#0F766E]">
            Simulation inputs
          </div>
          <h2 className="mt-2 text-xl font-black text-[#0F172A]">
            Configure scenario
          </h2>

          <div className="mt-6 space-y-4">
            {[
              ["patientsPerDay", "Patients / day"],
              ["acquisitionRate", "Acquisition rate"],
              ["bandwidthMbps", "Bandwidth (Mbps)"],
              ["reviewCapacity", "Review capacity"],
            ].map(([field, label]) => (
              <div key={field}>
                <label className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#64748B]">
                  {label}
                </label>
                <input
                  type="number"
                  min="0"
                  value={form[field]}
                  onChange={(e) => updateField(field, e.target.value)}
                  className="mt-2 w-full rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] px-4 py-3 text-[12px] font-medium text-[#334155] outline-none focus:border-[#A9D5C4] focus:bg-white"
                />
              </div>
            ))}
          </div>

          <button
            type="button"
            onClick={runSimulation}
            disabled={loading}
            className="mt-6 inline-flex w-full items-center justify-center gap-3 rounded-full bg-[#111816] px-5 py-3.5 text-[11px] font-bold text-white transition hover:bg-[#1C2723] disabled:opacity-50"
          >
            {loading ? (
              <>
                <RefreshCcw size={14} className="animate-spin" />
                Running simulation
              </>
            ) : (
              <>
                <Play size={14} />
                Run simulation
                <ArrowRight size={14} />
              </>
            )}
          </button>

          {!SIMULATION_URL && (
            <div className="mt-4 text-[10px] leading-5 text-[#64748B]">
              No Simulink endpoint is connected. Local capacity check is used.
            </div>
          )}
        </div>

        <div className="rounded-[30px] bg-[#111816] p-5 shadow-xl">
          <div className="rounded-[24px] border border-[#2B3B35] bg-[#16201D] p-6">
            <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#80958B]">
              Simulation output
            </div>
            <h2 className="mt-3 text-2xl font-black text-white">
              District workflow model
            </h2>

            {error ? (
              <div className="mt-6 rounded-2xl border border-[#78350F] bg-[#451A03] p-4 text-[11px] text-[#FCD34D]">
                {error}
              </div>
            ) : result ? (
              <div className="mt-6 space-y-4">
                {!SIMULATION_URL && (
                  <div className="rounded-2xl border border-[#A7CBB9] bg-[#183128] p-4 text-[11px] text-[#DCE8E1]">
                    <div className="flex items-center gap-2 font-bold">
                      <CheckCircle2 size={14} className="text-[#6EE7B7]" />
                      Local scenario check completed
                    </div>
                    <div className="mt-2 text-[#A9BEB4]">
                      No Simulink output is being claimed.
                    </div>
                  </div>
                )}

                <div className="grid gap-3 md:grid-cols-3">
                  <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723] p-4">
                    <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#80958B]">
                      Patients / day
                    </div>
                    <div className="mt-3 text-2xl font-black text-white">
                      {result.inputs.patientsPerDay}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723] p-4">
                    <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#80958B]">
                      Review utilization
                    </div>
                    <div className="mt-3 text-2xl font-black text-white">
                      {result.derived.reviewCapacityUtilizationPercent}%
                    </div>
                  </div>

                  <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723] p-4">
                    <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#80958B]">
                      Capacity gap / day
                    </div>
                    <div className="mt-3 text-2xl font-black text-white">
                      {result.derived.reviewCapacityGapPerDay}
                    </div>
                  </div>
                </div>

                <div className="rounded-2xl border border-[#2B3B35] bg-[#1C2723] p-5">
                  <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#80958B]">
                    Scenario status
                  </div>
                  <div className="mt-3 text-lg font-black text-white">
                    {result.derived.capacityStatus === "WITHIN_REVIEW_CAPACITY"
                      ? "Within review capacity"
                      : "Review capacity exceeded"}
                  </div>
                  <div className="mt-2 text-[11px] leading-5 text-[#A9BEB4]">
                    Acquisition rate: {result.inputs.acquisitionRate} ·
                    Bandwidth: {result.inputs.bandwidthMbps} Mbps · Review
                    capacity: {result.inputs.reviewCapacity}/day
                  </div>
                </div>

                <pre className="max-h-[230px] overflow-auto rounded-2xl border border-[#2B3B35] bg-[#0F1714] p-4 text-[10px] leading-5 text-[#B9CCC2]">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            ) : (
              <div className="mt-6 flex min-h-[360px] items-center justify-center rounded-2xl border border-[#2B3B35] bg-[#1C2723] text-center">
                <div>
                  <Network size={32} className="mx-auto text-[#80958B]" />
                  <div className="mt-4 text-[13px] font-bold text-[#DCE8E1]">
                    No simulation result
                  </div>
                  <div className="mt-2 max-w-[390px] text-[11px] leading-5 text-[#80958B]">
                    Enter the scenario inputs and run the connected simulation
                    or local capacity check.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </section>
    </PageShell>
  );
}
