import { useCallback, useEffect, useMemo, useState } from "react";

import {
  Activity,
  BarChart3,
  CheckCircle2,
  RefreshCcw,
} from "lucide-react";

import {
  Bar,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import PageShell from "../../components/layout/PageShell";

const ANALYTICS_URL = import.meta.env.VITE_ANALYTICS_URL || "";

function readLocalReviews() {
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
        if (review?.caseId) records.push(review);
      } catch {
        // Ignore malformed local records.
      }
    }
  } catch {
    // Ignore localStorage access errors.
  }

  return records;
}

function buildLocalAnalytics(records) {
  const screenings = records.length;
  const referrals = records.filter(
    (item) => item?.decision === "refer"
  ).length;
  const reviews = records.filter(
    (item) => Boolean(item?.decision)
  ).length;

  // Severity is shown only when a completed review actually persisted a grade.
  const gradeCounts = new Map();

  records.forEach((item) => {
    const grade =
      item?.grade ??
      item?.icdrGrade ??
      item?.icdr_grade ??
      null;

    if (grade === null || grade === undefined || grade === "") return;

    const numericGrade = Number(grade);
    if (!Number.isFinite(numericGrade)) return;

    const label =
      numericGrade === 0
        ? "Grade 0"
        : numericGrade === 1
          ? "Grade 1"
          : numericGrade === 2
            ? "Grade 2"
            : numericGrade === 3
              ? "Grade 3"
              : numericGrade === 4
                ? "Grade 4"
                : `Grade ${numericGrade}`;

    gradeCounts.set(label, (gradeCounts.get(label) || 0) + 1);
  });

  const severity = Array.from(gradeCounts, ([label, value]) => ({
    label,
    value,
  }));

  return {
    source: "local-review-records",
    screenings,
    referrals,
    reviews,
    trend:
      screenings > 0
        ? [{ label: "Current session", value: screenings }]
        : [],
    severity,
  };
}

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [source, setSource] = useState("none");

  const fetchAnalytics = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      if (ANALYTICS_URL) {
        const response = await fetch(ANALYTICS_URL, {
          headers: {
            Accept: "application/json",
          },
        });

        if (!response.ok) {
          throw new Error(
            `Analytics request failed (${response.status}).`
          );
        }

        setData(await response.json());
        setSource("api");
      } else {
        setData(buildLocalAnalytics(readLocalReviews()));
        setSource("local");
      }
    } catch (err) {
      const local = buildLocalAnalytics(readLocalReviews());

      if (local.screenings > 0 || local.reviews > 0) {
        setData(local);
        setSource("local");
        setError("");
      } else {
        setData(null);
        setSource("none");
        setError(
          err?.message || "Unable to load analytics."
        );
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchAnalytics();

    const refreshLocal = () => {
      if (!ANALYTICS_URL) {
        setData(buildLocalAnalytics(readLocalReviews()));
        setSource("local");
      }
    };

    window.addEventListener("storage", refreshLocal);
    return () =>
      window.removeEventListener("storage", refreshLocal);
  }, [fetchAnalytics]);

  const trendData = useMemo(() => {
    if (Array.isArray(data?.trend)) return data.trend;
    if (Array.isArray(data?.timeSeries)) return data.timeSeries;
    return [];
  }, [data]);

  const severityData = useMemo(() => {
    if (Array.isArray(data?.severity)) return data.severity;
    if (Array.isArray(data?.severityDistribution)) {
      return data.severityDistribution;
    }
    return [];
  }, [data]);

  return (
    <PageShell
      eyebrow="Screening intelligence"
      title="Understand the"
      accentTitle="screening network."
      description="View analytics returned by a connected service or, when no analytics service is configured, metrics derived strictly from completed VisionX review records saved in this browser."
      rightContent={
        <button
          type="button"
          onClick={fetchAnalytics}
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
      {error ? (
        <div className="mt-8 rounded-2xl border border-[#F1D2D0] bg-[#FFF7F7] p-5 text-[11px] text-[#9A5F5B]">
          {error}
        </div>
      ) : null}

      {loading ? (
        <div className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
          <RefreshCcw
            size={28}
            className="mx-auto animate-spin text-[#0F766E]"
          />
          <div className="mt-4 text-[13px] font-bold text-[#334155]">
            Loading analytics
          </div>
        </div>
      ) : (
        <>
          {source === "local" && (
            <div className="mt-8 inline-flex items-center gap-2 rounded-full border border-[#D1FAE5] bg-[#ECFDF5] px-3 py-1.5 text-[9px] font-bold uppercase tracking-[0.12em] text-[#047857]">
              <CheckCircle2 size={11} />
              Demo-session analytics
            </div>
          )}

          {!data ? (
            <div className="mt-5 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
              <BarChart3
                size={32}
                className="mx-auto text-[#CBD5E1]"
              />
              <div className="mt-4 text-[14px] font-bold text-[#334155]">
                No analytics data available
              </div>
            </div>
          ) : (
            <>
              <section className="mt-5 grid gap-4 md:grid-cols-3">
                {[
                  [
                    "Screenings",
                    data?.screenings ??
                      data?.totalScreenings ??
                      "—",
                  ],
                  [
                    "Referrals",
                    data?.referrals ??
                      data?.totalReferrals ??
                      "—",
                  ],
                  [
                    "Review volume",
                    data?.reviews ??
                      data?.reviewVolume ??
                      "—",
                  ],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm"
                  >
                    <div className="text-[9px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                      {label}
                    </div>
                    <div className="mt-3 text-[28px] font-black text-[#0F172A]">
                      {value}
                    </div>
                  </div>
                ))}
              </section>

              <div className="mt-6 grid gap-6 xl:grid-cols-2">
                <section className="rounded-[28px] border border-[#E2E8F0] bg-white p-6">
                  <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                      <Activity size={17} />
                    </div>

                    <div>
                      <div className="text-[10px] font-bold uppercase tracking-[0.20em] text-[#0F766E]">
                        Screening trend
                      </div>
                      <h2 className="mt-1 text-lg font-black text-[#0F172A]">
                        Activity over time
                      </h2>
                    </div>
                  </div>

                  <div className="mt-6 h-[300px]">
                    {trendData.length === 0 ? (
                      <div className="flex h-full items-center justify-center rounded-2xl bg-[#F8FAFC] text-[11px] text-[#64748B]">
                        No time-series data returned.
                      </div>
                    ) : (
                      <ResponsiveContainer
                        width="100%"
                        height="100%"
                      >
                        <LineChart data={trendData}>
                          <CartesianGrid stroke="#E2E8F0" />
                          <XAxis dataKey="label" />
                          <YAxis />
                          <Tooltip />
                          <Line
                            type="monotone"
                            dataKey="value"
                            stroke="#0F766E"
                            strokeWidth={2}
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </section>

                <section className="rounded-[28px] border border-[#E2E8F0] bg-white p-6">
                  <div className="text-[10px] font-bold uppercase tracking-[0.20em] text-[#0F766E]">
                    Severity distribution
                  </div>

                  <h2 className="mt-2 text-lg font-black text-[#0F172A]">
                    Returned model categories
                  </h2>

                  <div className="mt-6 h-[300px]">
                    {severityData.length === 0 ? (
                      <div className="flex h-full items-center justify-center rounded-2xl bg-[#F8FAFC] text-[11px] text-[#64748B]">
                        Severity data not persisted in the current review records.
                      </div>
                    ) : (
                      <ResponsiveContainer
                        width="100%"
                        height="100%"
                      >
                        <BarChart data={severityData}>
                          <CartesianGrid stroke="#E2E8F0" />
                          <XAxis dataKey="label" />
                          <YAxis />
                          <Tooltip />
                          <Bar
                            dataKey="value"
                            fill="#0F766E"
                            radius={[6, 6, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    )}
                  </div>
                </section>
              </div>

              {source === "local" && (
                <div className="mt-6 rounded-2xl border border-[#E2E8F0] bg-white p-5 text-[11px] leading-5 text-[#64748B]">
                  These metrics are derived only from completed review records
                  saved in this browser. They are not district-wide or
                  production analytics.
                </div>
              )}
            </>
          )}
        </>
      )}
    </PageShell>
  );
}
