import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { Navbar } from "./components/Navbar";
import { HomePage } from "./pages/HomePage";
import { UploadPage } from "./pages/UploadPage";
import { SearchPage } from "./pages/SearchPage";
import { CandidatesPage } from "./pages/CandidatesPage";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50/30 to-purple-50/30">
        <div className="absolute inset-0 bg-grid-pattern opacity-[0.02] pointer-events-none"></div>
        <Navbar />
        <main className="container mx-auto px-4 py-8 relative z-10">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/search" element={<SearchPage />} />
            <Route path="/candidates" element={<CandidatesPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
