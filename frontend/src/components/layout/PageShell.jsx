import { ArrowLeft, ArrowRight, Eye } from "lucide-react";
import { Link } from "react-router-dom";

export default function PageShell({
  eyebrow,
  title,
  accentTitle,
  description,
  rightContent,
  children,
}) {
  return (
    <main className="min-h-screen bg-[#F8FAFC] text-[#0F172A]">
      {/* Background grid */}
      <div
        className="pointer-events-none fixed inset-0 opacity-[0.35]"
        style={{
          backgroundImage:
            "linear-gradient(#E2E8F0 1px, transparent 1px), linear-gradient(90deg, #E2E8F0 1px, transparent 1px)",
          backgroundSize: "72px 72px",
        }}
      />

      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-[#E2E8F0] bg-white/95 backdrop-blur-md">
          <div className="mx-auto flex max-w-[1460px] items-center justify-between px-6 py-5 md:px-10 lg:px-12">
            <Link
              to="/"
              className="group flex items-center gap-3"
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-[#E2E8F0] bg-white text-[#0F766E] shadow-sm transition group-hover:border-[#CBD5E1]">
                <Eye size={20} />
              </div>

              <div>
                <div className="text-[17px] font-black tracking-[0.29em] text-[#0F172A]">
                  VISIONX
                </div>

                <div className="mt-0.5 text-[9px] font-bold uppercase tracking-[0.22em] text-[#64748B]">
                  Retinal Intelligence System
                </div>
              </div>
            </Link>

            <div className="hidden items-center gap-7 text-[11px] font-bold text-[#64748B] lg:flex">
              <Link to="/screening" className="transition hover:text-[#0F172A]">
                Screening
              </Link>

              <Link to="/doctor" className="transition hover:text-[#0F172A]">
                Clinical Review
              </Link>

              <Link to="/patients" className="transition hover:text-[#0F172A]">
                Patients
              </Link>

              <Link to="/referrals" className="transition hover:text-[#0F172A]">
                Referrals
              </Link>

              <Link to="/phc" className="transition hover:text-[#0F172A]">
                PHC
              </Link>

              <Link to="/analytics" className="transition hover:text-[#0F172A]">
                Analytics
              </Link>
            </div>

            <Link
              to="/"
              className="inline-flex items-center gap-2 rounded-full border border-[#E2E8F0] bg-white px-4 py-2.5 text-[10px] font-bold text-[#64748B] transition hover:border-[#CBD5E1] hover:text-[#334155]"
            >
              <ArrowLeft size={13} />
              Home
            </Link>
          </div>
        </header>

        {/* Page */}
        <div className="mx-auto max-w-[1460px] px-6 py-10 md:px-10 lg:px-12">
          <section className="flex flex-col justify-between gap-7 lg:flex-row lg:items-end">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-[0.30em] text-[#0F766E]">
                {eyebrow}
              </div>

              <h1 className="mt-4 text-[42px] font-black leading-none tracking-tight text-[#0F172A] md:text-[54px]">
                {title}

                {accentTitle ? (
                  <span className="block text-[#0F766E]">
                    {accentTitle}
                  </span>
                ) : null}
              </h1>

              <p className="mt-6 max-w-[720px] text-[16px] font-medium leading-relaxed text-[#475569]">
                {description}
              </p>
            </div>

            {rightContent}
          </section>

          {children}

          <footer className="mt-10 border-t border-[#E2E8F0] pt-6">
            <div className="flex flex-col justify-between gap-3 text-[10px] font-medium text-[#64748B] md:flex-row">
              <div>
                VisionX • Clinical screening platform
              </div>

              <div className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-[#059669]" />
                Backend-driven workspace
              </div>
            </div>
          </footer>
        </div>
      </div>
    </main>
  );
}