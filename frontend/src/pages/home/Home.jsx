import { motion } from "framer-motion";
import { Link } from "react-router-dom";

import {
  Activity,
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleDot,
  Database,
  Eye,
  FileCheck2,
  Hospital,
  Network,
  ScanLine,
  ShieldCheck,
  Stethoscope,
  UploadCloud,
  Users,
  Wifi,
} from "lucide-react";

import RetinaScene from "../../components/three/RetinaScene";

const COLORS = {
  background: "#F8FAFC", 
  forest: "#0F172A", 
  forest2: "#1E293B",
  forest3: "#334155",
  headingMuted: "#0F766E", 
  body: "#334155", 
  muted: "#64748B", 
  border: "#E2E8F0",
  soft: "#F1F5F9",
  accent: "#0F766E",
  success: "#059669", 
};

const modules = [
  {
    icon: ScanLine,
    title: "Screening",
    text: "Capture a retinal image and move through the complete screening workflow.",
    href: "/screening",
  },
  {
    icon: Stethoscope,
    title: "Clinical Review",
    text: "Review screening findings and validate the final case decision.",
    href: "/doctor",
  },
  {
    icon: Hospital,
    title: "PHC Operations",
    text: "Monitor screening activity, devices, connectivity and local workload.",
    href: "/phc",
  },
  {
    icon: BarChart3,
    title: "Analytics",
    text: "Understand screening volume, severity distribution and referrals.",
    href: "/analytics",
  },
  {
    icon: Network,
    title: "Simulation",
    text: "Model district-level throughput, bandwidth and specialist capacity.",
    href: "/simulation",
  },
  {
    icon: Database,
    title: "Validation",
    text: "Review model performance, datasets, calibration and benchmarks.",
    href: "/validation",
  },
];

const workflow = [
  {
    number: "01",
    title: "Acquire",
    description:
      "Capture or upload a retinal fundus image at the point of care.",
    icon: UploadCloud,
  },
  {
    number: "02",
    title: "Assess",
    description:
      "Evaluate image adequacy before downstream analysis.",
    icon: ScanLine,
  },
  {
    number: "03",
    title: "Analyse",
    description:
      "Generate disease severity and retinal evidence.",
    icon: BrainCircuit,
  },
  {
    number: "04",
    title: "Explain",
    description:
      "Expose the visual evidence supporting the model output.",
    icon: Eye,
  },
  {
    number: "05",
    title: "Review",
    description:
      "Keep the final case inside a human-in-the-loop workflow.",
    icon: ShieldCheck,
  },
];

function Pill({ children }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-2 text-[11px] font-bold uppercase tracking-[0.15em] text-[#334155] shadow-sm">
      <span className="h-1.5 w-1.5 rounded-full bg-[#059669]" />
      {children}
    </span>
  );
}

function Metric({ icon: Icon, value, label }) {
  return (
    <div className="group rounded-2xl border border-[#E2E8F0] bg-white p-5 shadow-sm transition-all duration-300 hover:-translate-y-0.5 hover:border-[#CBD5E1] hover:shadow-md">
      <div className="flex items-start justify-between">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-[#CCFBF1] bg-[#F0FDFA] text-[#0F766E]">
          <Icon size={18} strokeWidth={2} />
        </div>

        <CircleDot
          size={14}
          strokeWidth={2}
          className="text-[#94A3B8]"
        />
      </div>

      <div className="mt-7 text-2xl font-bold tracking-tight text-[#0F172A]">
        {value}
      </div>

      <div className="mt-1 text-[10px] font-bold uppercase tracking-[0.15em] text-[#64748B]">
        {label}
      </div>
    </div>
  );
}

function WorkflowCard({
  number,
  title,
  description,
  icon: Icon,
}) {
  return (
    <motion.div
      whileHover={{ y: -5 }}
      className="group relative overflow-hidden rounded-3xl border border-[#E2E8F0] bg-white p-6 shadow-sm transition-shadow duration-300 hover:shadow-md"
    >
      <div className="absolute left-0 right-0 top-0 h-1 bg-[#0F766E] opacity-0 transition duration-300 group-hover:opacity-100" />

      <div className="flex items-start justify-between">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-[#CCFBF1] bg-[#F0FDFA] text-[#0F766E]">
          <Icon size={20} strokeWidth={2} />
        </div>

        <span className="text-[11px] font-bold tracking-[0.18em] text-[#94A3B8]">
          {number}
        </span>
      </div>

      <h3 className="mt-12 text-lg font-bold text-[#0F172A]">
        {title}
      </h3>

      <p className="mt-3 text-sm leading-6 text-[#475569]">
        {description}
      </p>

      <div className="mt-7 h-px bg-[#F1F5F9]" />

      <div className="mt-4 flex items-center gap-2 text-[11px] font-bold uppercase tracking-[0.15em] text-[#64748B]">
        Workflow stage
        <ChevronRight size={14} />
      </div>
    </motion.div>
  );
}

function ScreeningPreview() {
  return (
    <div className="relative">
      <div className="rounded-[30px] border border-[#E2E8F0] bg-white p-3 shadow-lg">

        {/* Header */}
        <div className="flex items-center justify-between rounded-2xl border border-[#F1F5F9] bg-[#F8FAFC] px-4 py-3.5">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#0F766E]">
              <Eye size={18} strokeWidth={2} />
            </div>

            <div>
              <div className="text-[13px] font-bold text-[#0F172A]">
                New Screening
              </div>
              <div className="text-[11px] font-medium text-[#64748B]">
                PHC • Retinal Imaging
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-[#D1FAE5] bg-[#ECFDF5] px-3 py-1.5 text-[10px] font-bold uppercase tracking-[0.12em] text-[#047857]">
            <span className="h-2 w-2 rounded-full bg-[#059669]" />
            System Ready
          </div>
        </div>

        {/* Workspace */}
        <div className="mt-3 grid gap-3 lg:grid-cols-[1.27fr_0.73fr]">

          {/* Imaging viewer */}
          <div className="relative min-h-[410px] overflow-hidden rounded-2xl bg-[#020617]">

            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.025),transparent_55%)]" />

            <div
              className="absolute left-1/2 top-1/2 h-[315px] w-[315px] -translate-x-1/2 -translate-y-1/2 rounded-full"
              style={{
                background:
                  "radial-gradient(circle at 49% 47%, #A96855 0%, #7B4036 24%, #532C26 47%, #2A1917 75%, #151A19 100%)",
                boxShadow:
                  "0 0 90px rgba(163,92,71,0.13)",
              }}
            />

            <div className="absolute left-[65%] top-[35%] h-11 w-11 rounded-full bg-[#E6BA91]/60 blur-[5px]" />

            <div className="absolute left-[49%] top-[51%] h-5 w-5 rounded-full border border-[#DCA28E]/45" />

            {/* Vessel lines */}
            <div className="absolute left-[49%] top-[50%] h-[175px] w-[2px] origin-top rotate-[28deg] rounded-full bg-[#D08A78]/60" />
            <div className="absolute left-[50%] top-[50%] h-[160px] w-[1px] origin-top rotate-[-32deg] rounded-full bg-[#D98E7C]/55" />
            <div className="absolute left-[49%] top-[50%] h-[210px] w-[1px] origin-top rotate-[72deg] rounded-full bg-[#C97C6C]/40" />
            <div className="absolute left-[50%] top-[50%] h-[195px] w-[1px] origin-top rotate-[-68deg] rounded-full bg-[#C97C6C]/38" />

            {/* Evidence markers */}
            <div className="absolute left-[39%] top-[40%] h-3 w-3 rounded-full border-2 border-[#E4B65E] bg-[#E4B65E]/20" />

            <div className="absolute left-[58%] top-[55%] h-3 w-3 rounded-full border-2 border-[#B96B66] bg-[#B96B66]/20" />

            <div className="absolute left-[43%] top-[64%] h-2.5 w-2.5 rounded-full border border-[#E4B65E] bg-[#E4B65E]/20" />

            {/* Viewer info */}
            <div className="absolute left-4 top-4 rounded-xl border border-white/20 bg-black/50 px-3 py-2.5 backdrop-blur-md">
              <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                Fundus View
              </div>

              <div className="mt-1 text-[11px] font-medium text-white">
                Right Eye • OD
              </div>
            </div>

            {/* Viewer controls */}
            <div className="absolute bottom-4 left-4 flex gap-2">
              {["Original", "Evidence", "Vessels"].map(
                (label, index) => (
                  <button
                    key={label}
                    type="button"
                    className={`rounded-lg border px-3 py-2 text-[10px] font-medium transition ${
                      index === 0
                        ? "border-white/30 bg-white/20 text-white"
                        : "border-white/20 bg-black/40 text-gray-300 hover:bg-white/20"
                    }`}
                  >
                    {label}
                  </button>
                )
              )}
            </div>

            <div className="absolute right-4 top-4 flex flex-col gap-2">
              <button
                type="button"
                className="rounded-lg border border-white/20 bg-black/50 px-3 py-2 text-sm text-white backdrop-blur"
              >
                +
              </button>

              <button
                type="button"
                className="rounded-lg border border-white/20 bg-black/50 px-3 py-2 text-sm text-white backdrop-blur"
              >
                −
              </button>
            </div>

            <div className="absolute bottom-4 right-4 rounded-lg border border-white/20 bg-black/50 px-3 py-2 text-[10px] text-gray-300 backdrop-blur">
              Imaging channel • RGB
            </div>
          </div>

          {/* Clinical panel */}
          <div className="rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] p-5">

            <div className="text-[11px] font-bold uppercase tracking-[0.15em] text-[#64748B]">
              Screening overview
            </div>

            <div className="mt-5 rounded-xl border border-[#E2E8F0] bg-white p-4 shadow-sm">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-[12px] font-semibold text-[#475569]">
                    Image quality
                  </div>

                  <div className="mt-1 text-[17px] font-bold text-[#0F172A]">
                    Gradable
                  </div>
                </div>

                <div className="text-right">
                  <div className="text-[22px] font-bold text-[#0F766E]">
                    94%
                  </div>

                  <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-[#94A3B8]">
                    Quality index
                  </div>
                </div>
              </div>

              <div className="mt-4 h-2 overflow-hidden rounded-full bg-[#F1F5F9]">
                <div className="h-full w-[94%] rounded-full bg-[#0F766E]" />
              </div>
            </div>

            <div className="mt-4 space-y-2">
              {[
                ["Image quality", "Passed", "done"],
                ["Retinal analysis", "Ready", "ready"],
                ["Evidence map", "Available", "ready"],
                ["Clinical review", "Pending", "pending"],
              ].map(([label, status, state]) => (
                <div
                  key={label}
                  className="flex items-center justify-between rounded-xl border border-[#E2E8F0] bg-white px-4 py-3.5 shadow-sm"
                >
                  <div className="flex items-center gap-3">
                    {state === "done" ? (
                      <CheckCircle2
                        size={16}
                        className="text-[#059669]"
                      />
                    ) : (
                      <CircleDot
                        size={16}
                        strokeWidth={2}
                        className={
                          state === "ready"
                            ? "text-[#0F766E]"
                            : "text-[#CBD5E1]"
                        }
                      />
                    )}

                    <span className="text-[12px] font-bold text-[#334155]">
                      {label}
                    </span>
                  </div>

                  <span
                    className={`text-[11px] font-bold ${
                      state === "pending"
                        ? "text-[#94A3B8]"
                        : "text-[#0F172A]"
                    }`}
                  >
                    {status}
                  </span>
                </div>
              ))}
            </div>

            <button
              type="button"
              className="mt-5 flex w-full items-center justify-between rounded-xl bg-[#111816] px-5 py-4 text-[12px] font-bold text-white shadow-md transition hover:bg-[#1C2723]"
            >
              Continue to Analysis
              <ArrowRight size={16} />
            </button>
          </div>
        </div>

        {/* Case footer */}
        <div className="mt-3 flex items-center justify-between rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] px-5 py-3">
          <div className="flex items-center gap-5">
            <div className="text-[11px] font-bold uppercase tracking-[0.15em] text-[#64748B]">
              Case ID
              <span className="ml-2 font-bold text-[#0F172A]">
                VX-02481
              </span>
            </div>

            <div className="hidden h-3 w-px bg-[#CBD5E1] md:block" />

            <div className="hidden text-[11px] font-bold uppercase tracking-[0.15em] text-[#64748B] md:block">
              Device
              <span className="ml-2 font-bold text-[#0F172A]">
                PHC-CAM-04
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2 text-[11px] font-bold text-[#64748B]">
            <Network size={14} />
            Secure workflow
          </div>
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <main
      className="min-h-screen overflow-hidden bg-[#F8FAFC] text-[#0F172A]"
      style={{ backgroundColor: COLORS.background }}
    >

      {/* Background grid */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.6]"
        style={{
          backgroundImage:
            "linear-gradient(#E2E8F0 1px, transparent 1px), linear-gradient(90deg, #E2E8F0 1px, transparent 1px)",
          backgroundSize: "72px 72px",
        }}
      />

      {/* Soft atmospheric background */}
      <div className="pointer-events-none fixed right-[-220px] top-[-180px] h-[760px] w-[760px] rounded-full bg-[#E0F2FE] blur-[130px]" />

      <div className="relative z-10">

        {/* ================================================= */}
        {/* NAVBAR                                            */}
        {/* ================================================= */}

        <nav className="mx-auto flex max-w-[1460px] items-center justify-between px-6 py-6 md:px-10 lg:px-12">

          <Link
            to="/"
            className="group flex items-center gap-3"
          >
            <div className="flex h-12 w-12 items-center justify-center rounded-xl border border-[#CBD5E1] bg-white text-[#0F766E] shadow-sm transition group-hover:border-[#0D9488]">
              <Eye
                size={22}
                strokeWidth={2}
              />
            </div>

            <div>
              <div className="text-[18px] font-black tracking-[0.29em] text-[#0F172A]">
                VISIONX
              </div>

              <div className="mt-0.5 text-[9px] font-bold uppercase tracking-[0.30em] text-[#64748B]">
                Retinal Intelligence System
              </div>
            </div>
          </Link>

          <div className="hidden items-center gap-8 text-[13px] font-bold text-[#475569] xl:flex">

            {[
              ["Screening", "/screening"],
              ["Clinical Review", "/doctor"],
              ["PHC", "/phc"],
              ["Analytics", "/analytics"],
              ["Simulation", "/simulation"],
              ["Research", "/research"],
            ].map(([label, href]) => (
              <Link
                key={label}
                to={href}
                className="transition-colors duration-200 hover:text-[#0F172A]"
              >
                {label}
              </Link>
            ))}

          </div>

          <Link
            to="/screening"
            className="inline-flex items-center gap-2 rounded-full bg-[#111816] px-6 py-3.5 text-[12px] font-bold shadow-md transition hover:bg-[#1C2723]"
            style={{ color: "#FFFFFF" }}
          >
            Open Screening Console
            <ArrowRight size={15} />
          </Link>

        </nav>

        {/* ================================================= */}
        {/* HERO                                              */}
        {/* ================================================= */}

        <section className="mx-auto max-w-[1460px] px-6 pb-20 pt-12 md:px-10 lg:px-12 lg:pt-16">

          <div className="grid items-center gap-6 xl:grid-cols-[0.88fr_1.12fr]">

            {/* LEFT */}

            <motion.div
              initial={{
                opacity: 0,
                y: 24,
              }}
              animate={{
                opacity: 1,
                y: 0,
              }}
              transition={{
                duration: 0.75,
              }}
              className="relative z-20"
            >

              <Pill>
                Explainable retinal screening
              </Pill>

              <div className="mt-7 text-[11px] font-bold uppercase tracking-[0.30em] text-[#64748B]">
                Screen • Explain • Review • Refer
              </div>

              <h1
                className="mt-5 max-w-[700px] text-[56px] font-black leading-[0.95] tracking-tight sm:text-[67px] lg:text-[75px] xl:text-[86px]"
                style={{ color: COLORS.forest }}
              >
                See the retina.

                <span
                  className="mt-3 block"
                  style={{ color: COLORS.headingMuted }}
                >
                  Understand the evidence.
                </span>
              </h1>

              <p
                className="mt-8 max-w-[600px] text-[16px] leading-relaxed font-medium md:text-[18px]"
                style={{ color: COLORS.body }}
              >
                VisionX brings retinal image quality assessment, disease
                grading, visual evidence, clinical review and referral
                together in one connected screening platform.
              </p>

              <div className="mt-10 flex flex-wrap gap-4">

                <Link
                  to="/screening"
                  className="group inline-flex items-center gap-3 rounded-full bg-[#111816] px-7 py-4 text-[13px] font-bold shadow-lg transition duration-300 hover:-translate-y-0.5 hover:bg-[#1C2723]"
                  style={{ color: "#FFFFFF" }}
                >
                  Start a Screening

                  <ArrowRight
                    size={16}
                    className="transition-transform group-hover:translate-x-1"
                  />
                </Link>

                <a
                  href="#platform"
                  className="inline-flex items-center gap-2 rounded-full border-2 border-[#E2E8F0] bg-white px-7 py-4 text-[13px] font-bold text-[#334155] shadow-sm transition hover:-translate-y-0.5 hover:border-[#CBD5E1]"
                >
                  Explore Platform
                  <ChevronRight size={16} />
                </a>

              </div>

              <div className="mt-14 grid max-w-[590px] grid-cols-3 gap-4">

                <Metric
                  icon={Activity}
                  value="0–4"
                  label="ICDR grading"
                />

                <Metric
                  icon={BrainCircuit}
                  value="XAI"
                  label="Visual evidence"
                />

                <Metric
                  icon={ShieldCheck}
                  value="HITL"
                  label="Clinical review"
                />

              </div>

            </motion.div>

            {/* RIGHT — 3D */}

            <motion.div
              initial={{
                opacity: 0,
                x: 25,
                scale: 0.96,
              }}
              animate={{
                opacity: 1,
                x: 0,
                scale: 1,
              }}
              transition={{
                duration: 0.95,
                delay: 0.1,
              }}
              className="relative h-[670px] lg:h-[730px]"
            >

              <div className="absolute inset-0">
                <RetinaScene />
              </div>

              <div className="absolute right-4 top-[15%] rounded-2xl border border-[#E2E8F0] bg-white/95 px-5 py-4 shadow-lg backdrop-blur-md">

                <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#64748B]">
                  Imaging core
                </div>

                <div className="mt-2 flex items-center gap-2 text-[12px] font-bold text-[#0F172A]">
                  <span className="h-2 w-2 rounded-full bg-[#059669]" />
                  System operational
                </div>

              </div>

              <div className="absolute bottom-[18%] left-[2%] rounded-2xl border border-[#E2E8F0] bg-white/95 p-5 shadow-lg backdrop-blur-md">

                <div className="text-[10px] font-bold uppercase tracking-[0.21em] text-[#64748B]">
                  Retinal analysis
                </div>

                <div className="mt-4 grid grid-cols-2 gap-x-8 gap-y-3">

                  <div>
                    <div className="text-[11px] font-bold text-[#94A3B8]">
                      Vessel map
                    </div>

                    <div className="mt-1 text-[13px] font-bold text-[#0F172A]">
                      Available
                    </div>
                  </div>

                  <div>
                    <div className="text-[11px] font-bold text-[#94A3B8]">
                      Evidence
                    </div>

                    <div className="mt-1 text-[13px] font-bold text-[#0F172A]">
                      Ready
                    </div>
                  </div>

                </div>

              </div>

              <div className="absolute bottom-[7%] right-[13%] flex items-center gap-3 text-[11px] font-bold uppercase tracking-[0.22em] text-[#64748B]">

                <span className="h-0.5 w-10 bg-[#CBD5E1]" />

                3D retinal field

              </div>

            </motion.div>

          </div>

        </section>

        {/* ================================================= */}
        {/* WORKFLOW STRIP                                   */}
        {/* ================================================= */}

        <section className="border-y border-[#E2E8F0] bg-white">

          <div className="mx-auto grid max-w-[1460px] grid-cols-2 lg:grid-cols-4">

            {[
              ["01", "Image Quality", "Quality before inference"],
              ["02", "Retinal Analysis", "Structured findings"],
              ["03", "Clinical Review", "Human validation"],
              ["04", "Referral", "Connected care"],
            ].map(([number, title, text], index) => (

              <div
                key={title}
                className={`px-6 py-8 lg:px-10 ${
                  index < 3
                    ? "border-r border-[#E2E8F0]"
                    : ""
                }`}
              >

                <div className="text-[11px] font-bold tracking-[0.23em] text-[#94A3B8]">
                  {number}
                </div>

                <div className="mt-2 text-[15px] font-bold text-[#0F172A]">
                  {title}
                </div>

                <div className="mt-1 text-[12px] font-medium text-[#64748B]">
                  {text}
                </div>

              </div>

            ))}

          </div>

        </section>

        {/* ================================================= */}
        {/* WORKFLOW                                         */}
        {/* ================================================= */}

        <section
          id="platform"
          className="mx-auto max-w-[1460px] px-6 py-28 md:px-10 lg:px-12"
        >

          <div className="max-w-[780px]">

            <div className="text-[11px] font-bold uppercase tracking-[0.3em] text-[#0F766E]">
              The screening journey
            </div>

            <h2 className="mt-4 text-4xl font-black tracking-tight text-[#0F172A] md:text-5xl">
              From image capture

              <span className="block text-[#0F766E]">
                to clinical decision.
              </span>
            </h2>

            <p className="mt-5 max-w-[680px] text-[16px] font-medium leading-relaxed text-[#475569]">
              Each stage is represented as part of one continuous workflow,
              rather than isolated tools.
            </p>

          </div>

          <div className="mt-14 grid gap-5 md:grid-cols-2 xl:grid-cols-5">

            {workflow.map((item) => (
              <WorkflowCard
                key={item.number}
                {...item}
              />
            ))}

          </div>

        </section>

        {/* ================================================= */}
        {/* EXPLAINABILITY                                   */}
        {/* ================================================= */}

        <section className="border-y border-[#E2E8F0] bg-white">

          <div className="mx-auto grid max-w-[1460px] items-center gap-12 px-6 py-28 md:px-10 lg:grid-cols-[0.9fr_1.1fr] lg:px-12">

            <div>

              <div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">
                Evidence layer
              </div>

              <h2 className="mt-4 text-4xl font-black tracking-tight text-[#0F172A] md:text-5xl">
                Don't just show

                <span className="block text-[#0F766E]">
                  the prediction.
                </span>
              </h2>

              <p className="mt-6 max-w-[540px] text-[16px] font-medium leading-relaxed text-[#475569]">
                VisionX is designed to expose the evidence around a screening
                result so that an operator or clinical reviewer can understand
                what the system is highlighting.
              </p>

              <div className="mt-10 space-y-4">

                {[
                  "Image quality assessment",
                  "Retinal structure and lesion evidence",
                  "Visual explanation",
                  "Confidence and review state",
                ].map((item) => (

                  <div
                    key={item}
                    className="flex items-center gap-3 text-[15px] font-bold text-[#334155]"
                  >
                    <CheckCircle2
                      size={20}
                      className="text-[#059669]"
                    />

                    {item}
                  </div>

                ))}

              </div>

              <Link
                to="/screening/analysis"
                className="mt-10 inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-[#F8FAFC] px-6 py-4 text-[13px] font-bold text-[#0F172A] transition hover:bg-[#F1F5F9]"
              >
                View analysis workflow
                <ArrowRight size={16} />
              </Link>

            </div>

            {/* Dark analysis panel */}

            <div className="relative overflow-hidden rounded-[30px] bg-[#111816] p-5 shadow-2xl">

              <div className="rounded-[22px] border border-[#2B3B35] bg-[#16201D] p-6">

                <div className="flex items-center justify-between">

                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#80958B]">
                      Explainability workspace
                    </div>

                    <div className="mt-2 text-[15px] font-bold text-white">
                      Right Eye • Screening Case VX-02481
                    </div>
                  </div>

                  <span className="rounded-full border border-[#064E3B] bg-[#022C22] px-4 py-2 text-[10px] font-bold text-[#34D399]">
                    Evidence available
                  </span>

                </div>

                <div className="mt-6 grid gap-5 md:grid-cols-[1.15fr_0.85fr]">

                  <div className="relative min-h-[320px] overflow-hidden rounded-2xl bg-[#17100E]">

                    <div
                      className="absolute left-1/2 top-1/2 h-[250px] w-[250px] -translate-x-1/2 -translate-y-1/2 rounded-full"
                      style={{
                        background:
                          "radial-gradient(circle at 48% 47%, #A76451 0%, #743C32 35%, #3B211D 75%, #191917 100%)",
                      }}
                    />

                    <div className="absolute left-[62%] top-[36%] h-8 w-8 rounded-full bg-[#D5AB84]/70 blur-[5px]" />

                    <div className="absolute left-[40%] top-[41%] h-3 w-3 rounded-full border-2 border-[#D7A553] bg-[#D7A553]/20 shadow-[0_0_0_5px_rgba(215,165,83,0.15)]" />

                    <div className="absolute left-[58%] top-[56%] h-3 w-3 rounded-full border-2 border-[#B76967] bg-[#B76967]/20 shadow-[0_0_0_5px_rgba(183,105,103,0.15)]" />

                    <div className="absolute bottom-4 left-4 rounded-lg border border-white/20 bg-black/60 px-3 py-2 text-[10px] font-bold text-gray-300 backdrop-blur">
                      Grad-CAM evidence view
                    </div>

                  </div>

                  <div className="space-y-4">

                    <div className="rounded-xl border border-[#2B3B35] bg-[#1C2723]/50 p-5">

                      <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#80958B]">
                        Screening result
                      </div>

                      <div className="mt-3 text-3xl font-black text-white">
                        Level 2
                      </div>

                      <div className="mt-1 text-[12px] font-bold text-[#A9BEB4]">
                        Referable DR
                      </div>

                    </div>

                    <div className="rounded-xl border border-[#2B3B35] bg-[#1C2723]/50 p-5">

                      <div className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#80958B]">
                        Evidence
                      </div>

                      <div className="mt-5 space-y-4">

                        {[
                          ["Image quality", "94%"],
                          ["Model confidence", "91%"],
                          ["Evidence map", "Ready"],
                        ].map(([label, value]) => (

                          <div
                            key={label}
                            className="flex items-center justify-between"
                          >

                            <span className="text-[12px] font-bold text-[#80958B]">
                              {label}
                            </span>

                            <span className="text-[12px] font-bold text-white">
                              {value}
                            </span>

                          </div>

                        ))}

                      </div>

                    </div>

                    <div className="rounded-xl border border-[#78350F] bg-[#451A03] p-4">

                      <div className="flex items-center gap-2 text-[12px] font-bold text-[#FCD34D]">
                        <ShieldCheck size={16} />
                        Human review required
                      </div>

                    </div>

                  </div>

                </div>

              </div>

            </div>

          </div>

        </section>

        {/* ================================================= */}
        {/* PHC                                              */}
        {/* ================================================= */}

        <section className="mx-auto max-w-[1460px] px-6 py-28 md:px-10 lg:px-12">

          <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">

            <div className="max-w-[780px]">

              <div className="text-[11px] font-bold uppercase tracking-[0.3em] text-[#0F766E]">
                Designed for distributed care
              </div>

              <h2 className="mt-4 text-4xl font-black tracking-tight text-[#0F172A] md:text-5xl">
                Built for the

                <span className="text-[#0F766E]">
                  {" "}
                  point of care.
                </span>
              </h2>

            </div>

            <p className="max-w-[430px] text-[16px] font-medium leading-relaxed text-[#475569]">
              The product experience should work equally well for a PHC
              operator, reviewing clinician and district-level coordinator.
            </p>

          </div>

          <div className="mt-14 grid gap-5 md:grid-cols-2 xl:grid-cols-4">

            {[
              {
                icon: Hospital,
                title: "PHC-ready workflow",
                text: "Move from image acquisition to referral without leaving the platform.",
              },
              {
                icon: Wifi,
                title: "Connectivity aware",
                text: "Represent offline operation and synchronisation as part of the system design.",
              },
              {
                icon: Users,
                title: "Human in the loop",
                text: "Escalate cases that require clinical validation instead of hiding the review step.",
              },
              {
                icon: Network,
                title: "District scale",
                text: "Connect local screening with specialist review and referral pathways.",
              },
            ].map(
              ({
                icon: Icon,
                title,
                text,
              }) => (

                <div
                  key={title}
                  className="rounded-3xl border border-[#E2E8F0] bg-white p-8 shadow-sm"
                >

                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                    <Icon size={20} strokeWidth={2} />
                  </div>

                  <h3 className="mt-10 text-[19px] font-bold text-[#0F172A]">
                    {title}
                  </h3>

                  <p className="mt-3 text-[15px] leading-relaxed text-[#475569]">
                    {text}
                  </p>

                </div>

              )
            )}

          </div>

        </section>

        {/* ================================================= */}
        {/* MODULES                                          */}
        {/* ================================================= */}

        <section className="border-y border-[#E2E8F0] bg-white">

          <div className="mx-auto max-w-[1460px] px-6 py-28 md:px-10 lg:px-12">

            <div className="max-w-[780px]">

              <div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">
                VisionX modules
              </div>

              <h2 className="mt-4 text-4xl font-black tracking-tight text-[#0F172A] md:text-5xl">
                One platform.

                <span className="block text-[#0F766E]">
                  Multiple clinical workspaces.
                </span>
              </h2>

            </div>

            <div className="mt-14 grid gap-5 md:grid-cols-2 xl:grid-cols-3">

              {modules.map(
                ({
                  icon: Icon,
                  title,
                  text,
                  href,
                }) => (

                  <Link
                    key={title}
                    to={href}
                    className="group rounded-3xl border border-[#E2E8F0] bg-[#F8FAFC] p-8 transition duration-300 hover:-translate-y-1 hover:border-[#CBD5E1] hover:bg-white hover:shadow-lg"
                  >

                    <div className="flex items-start justify-between">

                      <div className="flex h-14 w-14 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#0F766E] shadow-sm">
                        <Icon size={22} strokeWidth={2} />
                      </div>

                      <ArrowRight
                        size={20}
                        className="text-[#94A3B8] transition group-hover:translate-x-1 group-hover:text-[#0F766E]"
                      />

                    </div>

                    <h3 className="mt-12 text-[20px] font-bold text-[#0F172A]">
                      {title}
                    </h3>

                    <p className="mt-3 text-[15px] leading-relaxed text-[#475569]">
                      {text}
                    </p>

                  </Link>

                )
              )}

            </div>

          </div>

        </section>

        {/* ================================================= */}
        {/* RESEARCH                                         */}
        {/* ================================================= */}

        <section className="mx-auto max-w-[1460px] px-6 py-28 md:px-10 lg:px-12">

          <div className="grid gap-12 lg:grid-cols-[0.85fr_1.15fr] lg:items-center">

            <div>

              <div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">
                Research & validation
              </div>

              <h2 className="mt-4 text-4xl font-black tracking-tight text-[#0F172A] md:text-5xl">
                Built around

                <span className="block text-[#0F766E]">
                  measurable evidence.
                </span>
              </h2>

              <p className="mt-6 max-w-[540px] text-[16px] font-medium leading-relaxed text-[#475569]">
                The project workflow can expose datasets, evaluation metrics,
                calibration, external validation and engineering assumptions
                as first-class parts of the system.
              </p>

              <Link
                to="/research"
                className="mt-10 inline-flex items-center gap-3 rounded-full bg-[#111816] px-7 py-4 text-[13px] font-bold shadow-md transition hover:bg-[#1C2723]"
                style={{ color: "#FFFFFF" }}
              >
                Explore research
                <ArrowRight size={16} />
              </Link>

            </div>

            <div className="grid gap-5 sm:grid-cols-2">

              {[
                ["APTOS 2019", "DR severity classification"],
                ["IDRiD", "Lesion-level annotations"],
                ["DRIVE", "Retinal vessel data"],
                ["MESSIDOR-2", "External validation"],
              ].map(([name, description]) => (

                <div
                  key={name}
                  className="rounded-3xl border border-[#E2E8F0] bg-white p-8 shadow-sm"
                >

                  <div className="flex items-center justify-between">

                    <Database
                      size={20}
                      className="text-[#0F766E]"
                    />

                    <span className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#64748B]">
                      Dataset
                    </span>

                  </div>

                  <div className="mt-10 text-[18px] font-bold text-[#0F172A]">
                    {name}
                  </div>

                  <div className="mt-2 text-[14px] font-medium text-[#475569]">
                    {description}
                  </div>

                </div>

              ))}

            </div>

          </div>

        </section>

        {/* ================================================= */}
        {/* FINAL CTA                                        */}
        {/* ================================================= */}

        <section className="px-6 pb-24 md:px-10 lg:px-12">

          <div className="mx-auto max-w-[1380px] overflow-hidden rounded-[34px] bg-[#111816] px-8 py-20 text-center shadow-2xl md:px-14 md:py-24">

            <div className="mx-auto max-w-[760px]">

              <div className="text-[11px] font-bold uppercase tracking-[0.3em] text-[#0D9488]">
                VisionX screening platform
              </div>

              <h2 className="mt-6 text-4xl font-black tracking-tight text-white md:text-6xl">
                Start with the retina.

                <span className="block text-[#80958B]">
                  Finish with evidence.
                </span>
              </h2>

              <p className="mx-auto mt-8 max-w-[610px] text-[16px] font-medium leading-relaxed text-[#A9BEB4]">
                Move from retinal image acquisition to explainable analysis,
                clinical review and referral through one connected workflow.
              </p>

              <div className="mt-10 flex flex-wrap justify-center gap-4">

                <Link
                  to="/screening"
                  className="inline-flex items-center gap-3 rounded-full bg-white px-8 py-4 text-[13px] font-bold shadow-lg transition hover:bg-[#F8FAFC]"
                  style={{ color: "#111816" }}
                >
                  Start Screening
                  <ArrowRight size={16} />
                </Link>

                <Link
                  to="/research"
                  className="inline-flex items-center gap-2 rounded-full border-2 border-[#2B3B35] bg-[#1C2723] px-8 py-4 text-[13px] font-bold transition hover:bg-[#2B3B35]"
                  style={{ color: "#FFFFFF" }}
                >
                  Research & Validation
                  <ChevronRight size={16} />
                </Link>

              </div>

            </div>

          </div>

        </section>

        {/* ================================================= */}
        {/* FOOTER                                           */}
        {/* ================================================= */}

        <footer className="border-t border-[#E2E8F0] bg-white">

          <div className="mx-auto flex max-w-[1460px] flex-col gap-6 px-6 py-10 md:flex-row md:items-center md:justify-between md:px-10 lg:px-12">

            <div>

              <div className="text-[16px] font-black tracking-[0.28em] text-[#0F172A]">
                VISIONX
              </div>

              <div className="mt-1 text-[10px] font-bold uppercase tracking-[0.22em] text-[#64748B]">
                Explainable retinal screening
              </div>

            </div>

            <div className="flex flex-wrap items-center gap-6 text-[11px] font-bold uppercase tracking-[0.15em] text-[#475569]">

              <Link
                to="/screening"
                className="hover:text-[#0F766E] transition"
              >
                Screening
              </Link>

              <Link
                to="/doctor"
                className="hover:text-[#0F766E] transition"
              >
                Clinical Review
              </Link>

              <Link
                to="/analytics"
                className="hover:text-[#0F766E] transition"
              >
                Analytics
              </Link>

              <Link
                to="/research"
                className="hover:text-[#0F766E] transition"
              >
                Research
              </Link>

            </div>

            <div className="flex items-center gap-2 text-[10px] font-bold text-[#64748B]">
              <FileCheck2 size={14} />
              SIH 2026 • MedTech / HealthTech
            </div>

          </div>

        </footer>

      </div>
    </main>
  );
}