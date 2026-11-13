import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import {
  ArrowRight,
  Eye,
  FileCheck2,
  RefreshCcw,
  Search,
  Users,
} from "lucide-react";

import PageShell from "../../components/layout/PageShell";

const PATIENTS_URL =
  import.meta.env.VITE_PATIENTS_URL || "";

function extractPatients(payload) {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.patients)) return payload.patients;
  if (Array.isArray(payload?.data)) return payload.data;
  if (Array.isArray(payload?.results)) return payload.results;

  return [];
}


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
        if (!review?.caseId) continue;

        records.push({
          id: review.patientId ?? review.patient_id ?? review.caseId,
          name: review.patientName ?? review.patient_name ?? "Not provided",
          age: review.age ?? null,
          sex: review.sex ?? review.gender ?? null,
          location: review.location ?? review.phc ?? null,
          lastScreening: review.timestamp ?? null,
          caseId: review.caseId,
          decision: review.decision ?? null,
          source: "local-review-record",
          raw: review,
        });
      } catch {
        // Ignore malformed local records.
      }
    }
  } catch {
    // Ignore localStorage access errors.
  }

  return records;
}

function normalizePatient(item, index) {
  return {
    key:
      item?.id ??
      item?.patientId ??
      item?.patient_id ??
      `row-${index}`,

    id:
      item?.id ??
      item?.patientId ??
      item?.patient_id ??
      "—",

    name:
      item?.name ??
      item?.patientName ??
      item?.patient_name ??
      "—",

    age:
      item?.age ??
      "—",

    sex:
      item?.sex ??
      item?.gender ??
      "—",

    location:
      item?.location ??
      item?.phc ??
      item?.phcName ??
      item?.phc_name ??
      "—",

    lastScreening:
      item?.lastScreening ??
      item?.last_screening ??
      item?.updatedAt ??
      item?.updated_at ??
      "—",
    caseId: item?.caseId ?? item?.case_id ?? null,
    source: item?.source ?? "api",
    raw: item,
  };
}

export default function Patients() {
  const navigate = useNavigate();

  const [patients, setPatients] = useState([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchPatients = useCallback(async () => {
    setLoading(true);
    setError("");

    try {
      if (PATIENTS_URL) {
        const response = await fetch(PATIENTS_URL, {
          headers: {
            Accept: "application/json",
          },
        });

        if (!response.ok) {
          throw new Error(
            `Patient request failed (${response.status}).`
          );
        }

        const payload = await response.json();
        setPatients(
          extractPatients(payload).map(normalizePatient)
        );
      } else {
        setPatients(readLocalReviews().map(normalizePatient));
      }
    } catch (err) {
      const local = readLocalReviews().map(normalizePatient);

      if (local.length > 0) {
        setPatients(local);
        setError("");
      } else {
        setPatients([]);
        setError(
          err?.message || "Unable to load patient records."
        );
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchPatients();
  }, [fetchPatients]);

  const filteredPatients = useMemo(() => {
    const query = search.trim().toLowerCase();

    if (!query) return patients;

    return patients.filter((item) =>
      [
        item.id,
        item.name,
        item.location,
      ]
        .join(" ")
        .toLowerCase()
        .includes(query)
    );
  }, [patients, search]);

  return (
    <PageShell
      eyebrow="Patient workspace"
      title="Manage the"
      accentTitle="patient record."
      description="Access patient records returned by the connected VisionX patient service and open individual screening histories."
      rightContent={
        <div className="flex items-center gap-3 rounded-2xl border border-[#E2E8F0] bg-white px-5 py-4 shadow-sm">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
            <Users size={16} />
          </div>

          <div>
            <div className="text-[9px] font-bold uppercase tracking-[0.17em] text-[#94A3B8]">
              Patient records
            </div>

            <div className="mt-1 text-[12px] font-bold text-[#334155]">
              {PATIENTS_URL ? "API driven" : "Demo-session records"}
            </div>
          </div>
        </div>
      }
    >
      <section className="mt-8 flex flex-col justify-between gap-4 md:flex-row md:items-center">
        <div>
          <div className="text-[10px] font-bold uppercase tracking-[0.24em] text-[#0F766E]">
            Records
          </div>

          <h2 className="mt-2 text-2xl font-black text-[#0F172A]">
            Patient directory
          </h2>
        </div>

        <div className="flex gap-2">
          <div className="relative">
            <Search
              size={15}
              className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#94A3B8]"
            />

            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search patient"
              className="w-full rounded-xl border border-[#E2E8F0] bg-white py-3 pl-10 pr-4 text-[11px] font-medium text-[#334155] outline-none placeholder:text-[#94A3B8] focus:border-[#A9D5C4] sm:w-[250px]"
            />
          </div>

          <button
            type="button"
            onClick={fetchPatients}
            disabled={loading}
            className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#64748B] transition hover:border-[#CBD5E1] disabled:opacity-40"
          >
            <RefreshCcw
              size={15}
              className={loading ? "animate-spin" : ""}
            />
          </button>
        </div>
      </section>

      {error ? (
        <div className="mt-5 rounded-2xl border border-[#F1D2D0] bg-[#FFF7F7] p-4 text-[11px] text-[#9A5F5B]">
          {error}
        </div>
      ) : null}

      {!PATIENTS_URL && patients.length === 0 ? (
        <div className="mt-5 rounded-[28px] border border-[#E2E8F0] bg-white px-6 py-16 text-center">
          <Users
            size={28}
            className="mx-auto text-[#CBD5E1]"
          />

          <h3 className="mt-4 text-[15px] font-bold text-[#334155]">
            Patient service not connected
          </h3>

          <p className="mx-auto mt-2 max-w-[520px] text-[11px] leading-5 text-[#64748B]">
            Configure VITE_PATIENTS_URL to load real patient
            records.
          </p>
        </div>
      ) : loading ? (
        <div className="mt-5 rounded-[28px] border border-[#E2E8F0] bg-white px-6 py-16 text-center">
          <RefreshCcw
            size={27}
            className="mx-auto animate-spin text-[#0F766E]"
          />

          <div className="mt-4 text-[13px] font-bold text-[#334155]">
            Loading patient records
          </div>
        </div>
      ) : filteredPatients.length === 0 ? (
        <div className="mt-5 rounded-[28px] border border-[#E2E8F0] bg-white px-6 py-16 text-center">
          <FileCheck2
            size={28}
            className="mx-auto text-[#CBD5E1]"
          />

          <h3 className="mt-4 text-[14px] font-bold text-[#334155]">
            No patient records returned
          </h3>

          <p className="mt-2 text-[11px] text-[#64748B]">
            The connected service returned no matching records.
          </p>
        </div>
      ) : (
        <div className="mt-5 space-y-3">
          {filteredPatients.map((patient) => (
            <div
              key={patient.key}
              className="rounded-2xl border border-[#E2E8F0] bg-white p-5 transition hover:border-[#CBD5E1] hover:shadow-sm"
            >
              <div className="grid gap-5 md:grid-cols-[1.2fr_0.7fr_0.9fr_0.9fr_auto] md:items-center">
                <div className="flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
                    <Eye size={17} />
                  </div>

                  <div>
                    <div className="text-[12px] font-bold text-[#0F172A]">
                      {patient.name}
                    </div>

                    <div className="mt-1 text-[10px] text-[#64748B]">
                      {patient.id}
                      {patient.caseId ? ` • ${patient.caseId}` : ""}
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
                    Age
                  </div>

                  <div className="mt-2 text-[11px] font-bold text-[#334155]">
                    {patient.age}
                  </div>
                </div>

                <div>
                  <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
                    Sex
                  </div>

                  <div className="mt-2 text-[11px] font-bold text-[#334155]">
                    {patient.sex}
                  </div>
                </div>

                <div>
                  <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
                    Location
                  </div>

                  <div className="mt-2 text-[11px] font-bold text-[#334155]">
                    {patient.location}
                  </div>
                </div>

                <button
                  type="button"
                  disabled={patient.id === "—"}
                  onClick={() =>
                    navigate(
                      `/patients/${encodeURIComponent(patient.id)}`,
                      {
                        state: {
                          patient: patient,
                        },
                      }
                    )
                  }
                  className="inline-flex items-center justify-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-2.5 text-[10px] font-bold text-[#334155] hover:bg-[#F8FAFC] disabled:opacity-40"
                >
                  Open
                  <ArrowRight size={13} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </PageShell>
  );
}