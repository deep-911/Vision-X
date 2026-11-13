import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";

import {
  ArrowLeft,
  Eye,
  FileCheck2,
  RefreshCcw,
  ShieldCheck,
} from "lucide-react";

import PageShell from "../../components/layout/PageShell";

const PATIENT_DETAIL_URL =
  import.meta.env.VITE_PATIENT_DETAIL_URL || "";

export default function PatientDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const [patient, setPatient] = useState(
    location.state?.patient?.raw ||
      location.state?.patient ||
      null
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchPatient = useCallback(async () => {
    if (!PATIENT_DETAIL_URL || !id) return;

    setLoading(true);
    setError("");

    try {
      const url = PATIENT_DETAIL_URL.replace(
        ":id",
        encodeURIComponent(id)
      );

      const response = await fetch(url, {
        headers: {
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        throw new Error(
          `Patient detail request failed (${response.status}).`
        );
      }

      const payload = await response.json();

      setPatient(payload?.patient ?? payload?.data ?? payload);
    } catch (err) {
      setError(
        err?.message ||
          "Unable to load patient information."
      );
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    if (!patient) {
      fetchPatient();
    }
  }, [patient, fetchPatient]);

  return (
    <PageShell
      eyebrow="Patient detail"
      title="Patient"
      accentTitle="record."
      description="Review the patient information and connected screening history returned by the VisionX backend."
      rightContent={
        <button
          type="button"
          onClick={() => navigate("/patients")}
          className="inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-3 text-[10px] font-bold text-[#64748B]"
        >
          <ArrowLeft size={13} />
          Patients
        </button>
      }
    >
      {loading ? (
        <div className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
          <RefreshCcw
            size={28}
            className="mx-auto animate-spin text-[#0F766E]"
          />

          <div className="mt-4 text-[13px] font-bold text-[#334155]">
            Loading patient record
          </div>
        </div>
      ) : error ? (
        <div className="mt-8 rounded-[28px] border border-[#F1D2D0] bg-[#FFF7F7] p-6 text-[11px] text-[#9A5F5B]">
          {error}
        </div>
      ) : !patient ? (
        <div className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-16 text-center">
          <Eye
            size={30}
            className="mx-auto text-[#CBD5E1]"
          />

          <div className="mt-4 text-[14px] font-bold text-[#334155]">
            Patient record unavailable
          </div>

          <div className="mt-2 text-[11px] text-[#64748B]">
            No patient data is currently available.
          </div>

          <button
            type="button"
            onClick={fetchPatient}
            disabled={!PATIENT_DETAIL_URL}
            className="mt-6 rounded-full bg-[#111816] px-5 py-3 text-[11px] font-bold text-white disabled:opacity-40"
          >
            Reload record
          </button>
        </div>
      ) : (
        <>
          <section className="mt-8 grid gap-5 lg:grid-cols-3">
            {[
              [
                "Patient ID",
                patient.id ??
                  patient.patientId ??
                  patient.patient_id ??
                  id ??
                  "—",
              ],
              [
                "Name",
                patient.name ??
                  patient.patientName ??
                  patient.patient_name ??
                  "—",
              ],
              [
                "Location",
                patient.location ??
                  patient.phc ??
                  patient.phcName ??
                  patient.phc_name ??
                  "—",
              ],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm"
              >
                <div className="text-[9px] font-bold uppercase tracking-[0.16em] text-[#94A3B8]">
                  {label}
                </div>

                <div className="mt-3 text-[16px] font-black text-[#0F172A]">
                  {value}
                </div>
              </div>
            ))}
          </section>

          <section className="mt-6 grid gap-6 lg:grid-cols-[1fr_0.72fr]">
            <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
              <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#0F766E]">
                Patient information
              </div>

              <div className="mt-5 grid gap-3 sm:grid-cols-2">
                {[
                  ["Age", patient.age],
                  ["Sex", patient.sex ?? patient.gender],
                  ["Date of birth", patient.dateOfBirth ?? patient.date_of_birth],
                  ["Contact", patient.contact],
                  ["Address", patient.address],
                  ["Last screening", patient.lastScreening ?? patient.last_screening],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="rounded-xl bg-[#F8FAFC] px-4 py-3.5"
                  >
                    <div className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#94A3B8]">
                      {label}
                    </div>

                    <div className="mt-2 text-[11px] font-bold text-[#334155]">
                      {value ?? "—"}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <ShieldCheck size={19} />
              </div>

              <h3 className="mt-6 text-lg font-black text-[#0F172A]">
                Screening history
              </h3>

              <p className="mt-2 text-[11px] leading-5 text-[#64748B]">
                Historical screenings will appear here when
                returned by the connected patient/case service.
              </p>
            </div>
          </section>

          <section className="mt-6 rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm">
            <div className="flex items-center gap-3">
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
                <FileCheck2 size={18} />
              </div>

              <div>
                <div className="text-[10px] font-bold uppercase tracking-[0.20em] text-[#0F766E]">
                  Record
                </div>

                <h2 className="mt-1 text-xl font-black text-[#0F172A]">
                  Connected screening data
                </h2>
              </div>
            </div>

            <div className="mt-6 rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] p-6 text-[11px] leading-6 text-[#64748B]">
              No synthetic screening history is displayed.
              Connect the patient record service to populate
              real cases and clinical events.
            </div>
          </section>
        </>
      )}
    </PageShell>
  );
}