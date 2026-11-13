import { useState } from "react";

import {
  CheckCircle2,
  Database,
  Globe,
  Save,
  ShieldCheck,
  Wifi,
} from "lucide-react";

import PageShell from "../../components/layout/PageShell";

export default function Settings() {
  const [saved, setSaved] = useState(false);

  const [settings, setSettings] = useState({
    apiBaseUrl:
      import.meta.env.VITE_API_BASE_URL || "",
    screeningUrl:
      import.meta.env.VITE_SCREENING_URL || "",
    reviewUrl:
      import.meta.env.VITE_REVIEW_QUEUE_URL || "",
  });

  const update = (field, value) => {
    setSaved(false);

    setSettings((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const handleSave = () => {
    /*
      Browser apps cannot modify .env files.
      This button intentionally only acknowledges the UI state.

      Persist these values through your actual configuration
      system when backend deployment is connected.
    */

    setSaved(true);
  };

  return (
    <PageShell
      eyebrow="Platform configuration"
      title="Configure the"
      accentTitle="VisionX workspace."
      description="Manage frontend configuration values and workspace preferences. Runtime environment variables remain the source of truth for deployed builds."
      rightContent={
        <div className="flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-2.5 text-[10px] font-bold text-[#64748B]">
          <ShieldCheck
            size={13}
            className="text-[#059669]"
          />
          Configuration
        </div>
      }
    >
      <div className="mt-8 grid gap-6 lg:grid-cols-[1fr_0.7fr]">
        <section className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
          <div className="text-[10px] font-bold uppercase tracking-[0.23em] text-[#0F766E]">
            Runtime configuration
          </div>

          <h2 className="mt-2 text-xl font-black text-[#0F172A]">
            Service endpoints
          </h2>

          <div className="mt-6 space-y-5">
            {[
              [
                "apiBaseUrl",
                "API base URL",
                Globe,
              ],
              [
                "screeningUrl",
                "Screening service",
                Wifi,
              ],
              [
                "reviewUrl",
                "Review queue service",
                Database,
              ],
            ].map(([field, label, Icon]) => (
              <div key={field}>
                <label className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-[#64748B]">
                  <Icon size={13} />
                  {label}
                </label>

                <input
                  type="text"
                  value={settings[field]}
                  onChange={(e) =>
                    update(
                      field,
                      e.target.value
                    )
                  }
                  placeholder="Not configured"
                  className="mt-2 w-full rounded-xl border border-[#E2E8F0] bg-[#F8FAFC] px-4 py-3 text-[11px] font-medium text-[#334155] outline-none placeholder:text-[#94A3B8] focus:border-[#A9D5C4] focus:bg-white"
                />
              </div>
            ))}
          </div>

          <button
            type="button"
            onClick={handleSave}
            className="mt-7 inline-flex items-center gap-2 rounded-full bg-[#111816] px-6 py-3.5 text-[11px] font-bold text-white transition hover:bg-[#1C2723]"
          >
            <Save size={14} />
            Save configuration
          </button>

          {saved ? (
            <div className="mt-4 flex items-center gap-2 text-[10px] font-bold text-[#059669]">
              <CheckCircle2 size={14} />
              Configuration state acknowledged.
            </div>
          ) : null}
        </section>

        <aside className="space-y-5">
          <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
              <ShieldCheck size={19} />
            </div>

            <h3 className="mt-6 text-lg font-black text-[#0F172A]">
              Configuration source
            </h3>

            <p className="mt-2 text-[11px] leading-5 text-[#64748B]">
              Vite environment variables are read at build
              time. Changing a field here does not modify the
              deployment environment automatically.
            </p>
          </div>

          <div className="overflow-hidden rounded-[28px] bg-[#111816] p-5">
            <div className="rounded-[22px] border border-[#2B3B35] bg-[#1C2723] p-5">
              <div className="text-[10px] font-bold uppercase tracking-[0.20em] text-[#80958B]">
                Current runtime state
              </div>

              <div className="mt-5 space-y-3">
                {[
                  ["API base", settings.apiBaseUrl],
                  ["Screening", settings.screeningUrl],
                  ["Review queue", settings.reviewUrl],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="flex items-center justify-between gap-4"
                  >
                    <span className="text-[10px] font-bold text-[#A9BEB4]">
                      {label}
                    </span>

                    <span className="max-w-[220px] truncate text-[10px] font-medium text-white">
                      {value || "Not configured"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </PageShell>
  );
}