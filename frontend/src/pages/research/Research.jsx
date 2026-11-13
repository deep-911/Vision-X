import {
  BookOpen,
  BrainCircuit,
  Database,
  Eye,
  FileCheck2,
  Network,
  ScanLine,
  ShieldCheck,
} from "lucide-react";

import PageShell from "../../components/layout/PageShell";

const areas = [
  {
    icon: ScanLine,
    title: "Image processing",
    text: "Image quality assessment, enhancement and retinal image preparation.",
  },
  {
    icon: BrainCircuit,
    title: "Disease grading",
    text: "Retinal disease severity modelling and screening inference.",
  },
  {
    icon: Eye,
    title: "Explainability",
    text: "Visual explanation using Grad-CAM and retinal evidence.",
  },
  {
    icon: ShieldCheck,
    title: "Human-in-the-loop",
    text: "Clinical review remains part of the screening workflow.",
  },
  {
    icon: Network,
    title: "Simulation",
    text: "Telemedicine workflow and district-level system modelling.",
  },
  {
    icon: Database,
    title: "Validation",
    text: "Datasets, benchmarks, evaluation and model validation.",
  },
];

const references = [
  {
    author: "Gulshan et al.",
    year: "2016",
    title:
      "Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy in Retinal Fundus Photographs.",
  },
  {
    author: "Porwal et al.",
    year: "2018",
    title:
      "Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research.",
  },
  {
    author: "Selvaraju et al.",
    year: "2017",
    title:
      "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.",
  },
  {
    author: "Wong et al.",
    year: "2018",
    title:
      "Guidelines on Diabetic Eye Care: International Council of Ophthalmology recommendations for screening, follow-up, referral, and treatment.",
  },
];

export default function Research() {
  return (
    <PageShell
      eyebrow="Research & validation"
      title="The research"
      accentTitle="behind VisionX."
      description="A research workspace for understanding the methods, explainability layer, validation pathway and systems modelling behind the platform."
      rightContent={
        <div className="flex items-center gap-3 rounded-2xl border border-[#E2E8F0] bg-white px-5 py-4 shadow-sm">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
            <BookOpen size={16} />
          </div>

          <div>
            <div className="text-[9px] font-bold uppercase tracking-[0.17em] text-[#94A3B8]">
              Research
            </div>

            <div className="mt-1 text-[12px] font-bold text-[#334155]">
              VisionX platform
            </div>
          </div>
        </div>
      }
    >
      <section className="mt-8 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        {areas.map((item) => {
          const Icon = item.icon;

          return (
            <div
              key={item.title}
              className="rounded-3xl border border-[#E2E8F0] bg-white p-7 shadow-sm"
            >
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[#F0FDFA] text-[#0F766E]">
                <Icon size={20} />
              </div>

              <h3 className="mt-8 text-[18px] font-bold text-[#0F172A]">
                {item.title}
              </h3>

              <p className="mt-3 text-[13px] leading-6 text-[#475569]">
                {item.text}
              </p>
            </div>
          );
        })}
      </section>

      <section className="mt-8 overflow-hidden rounded-[30px] bg-[#111816] p-5 shadow-2xl md:p-6">
        <div className="rounded-[24px] border border-[#2B3B35] bg-[#16201D] p-6">
          <div className="text-[10px] font-bold uppercase tracking-[0.22em] text-[#80958B]">
            Technical approach
          </div>

          <h2 className="mt-3 text-2xl font-black text-white">
            From retinal image
            <span className="text-[#80958B]">
              {" "}to explainable screening.
            </span>
          </h2>

          <div className="mt-7 grid gap-3 md:grid-cols-4">
            {[
              ["01", "Acquire", "Retinal image"],
              ["02", "Analyse", "Clinical findings"],
              ["03", "Explain", "Visual evidence"],
              ["04", "Review", "Human validation"],
            ].map(([number, title, text]) => (
              <div
                key={number}
                className="rounded-2xl border border-[#2B3B35] bg-[#1C2723] p-5"
              >
                <div className="text-[10px] font-bold tracking-[0.18em] text-[#80958B]">
                  {number}
                </div>

                <div className="mt-5 text-[12px] font-bold text-white">
                  {title}
                </div>

                <div className="mt-1 text-[10px] text-[#80958B]">
                  {text}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mt-8 rounded-[28px] border border-[#E2E8F0] bg-white p-6 shadow-sm md:p-7">
        <div className="flex items-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-[#F1F5F9] text-[#0F766E]">
            <FileCheck2 size={18} />
          </div>

          <div>
            <div className="text-[10px] font-bold uppercase tracking-[0.20em] text-[#0F766E]">
              References
            </div>

            <h2 className="mt-1 text-xl font-black text-[#0F172A]">
              Research foundation
            </h2>
          </div>
        </div>

        <div className="mt-7 space-y-3">
          {references.map((reference) => (
            <article
              key={`${reference.author}-${reference.year}`}
              className="rounded-2xl border border-[#E2E8F0] bg-[#F8FAFC] p-5"
            >
              <div className="text-[10px] font-bold uppercase tracking-[0.13em] text-[#64748B]">
                {reference.author} • {reference.year}
              </div>

              <div className="mt-2 text-[12px] font-bold leading-5 text-[#334155]">
                {reference.title}
              </div>
            </article>
          ))}
        </div>
      </section>
    </PageShell>
  );
}