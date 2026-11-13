import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
} from "react-router-dom";

import Home from "./pages/home/Home";
import Screening from "./pages/screening/Screening";
import Analysis from "./pages/analysis/Analysis";
import Explainability from "./pages/analysis/Explainability";
import DoctorReview from "./pages/doctor/DoctorReview";
import Doctor from "./pages/doctor/Doctor";

import Patients from "./pages/patients/Patients";
import PatientDetail from "./pages/patients/PatientDetail";

import Referrals from "./pages/referrals/Referrals";

import PHC from "./pages/phc/PHC";

import Analytics from "./pages/analytics/Analytics";

import Simulation from "./pages/simulation/Simulation";

import Validation from "./pages/validation/Validation";

import Research from "./pages/research/Research";

import Settings from "./pages/setting/Setting";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />

        <Route
          path="/screening"
          element={<Screening />}
        />
        <Route path="/screening/analysis" element={<Analysis />} />
        <Route
          path="/screening/explainability"
          element={<Explainability />}
        />

        <Route
          path="/doctor/review/:id"
          element={<DoctorReview />}
        />


        <Route path="/doctor" element={<Doctor />} />

       <Route path="/patients" element={<Patients />} />

<Route
  path="/patients/:id"
  element={<PatientDetail />}
/>

<Route path="/referrals" element={<Referrals />} />

<Route path="/phc" element={<PHC />} />

<Route path="/analytics" element={<Analytics />} />

<Route path="/simulation" element={<Simulation />} />

<Route path="/validation" element={<Validation />} />

<Route path="/research" element={<Research />} />

<Route path="/settings" element={<Settings />} />

        <Route
          path="*"
          element={<Navigate to="/" replace />}
        />
      </Routes>
    </BrowserRouter>
  );
}