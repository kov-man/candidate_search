import React, { useState, useEffect } from "react";
import {
  User,
  Calendar,
  Tag,
  ChevronRight,
  Loader2,
  Trash2,
  RefreshCw,
} from "lucide-react";
import { apiService } from "../services/api";
import type { Candidate } from "../types";

export function CandidatesPage() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedCandidate, setSelectedCandidate] = useState<string | null>(
    null
  );
  const [candidateSkills, setCandidateSkills] = useState<{
    [key: string]: { name: string; confidence: number }[];
  }>({});
  const [deletingCandidates, setDeletingCandidates] = useState<Set<string>>(
    new Set()
  );

  useEffect(() => {
    loadCandidates();
  }, []);

  const loadCandidates = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await apiService.listCandidates(0, 50);
      setCandidates(response.candidates);
    } catch (err: any) {
      setError("Failed to load candidates");
    } finally {
      setLoading(false);
    }
  };

  const loadCandidateSkills = async (candidateId: string) => {
    if (candidateSkills[candidateId]) return;

    try {
      // Get the candidate data which already includes skills
      const candidate = candidates.find((c) => c.candidate_id === candidateId);
      if (candidate && candidate.skills) {
        // Convert string skills to the expected format
        const skills = candidate.skills.map((skill) => ({
          name: skill,
          confidence: 1.0, // Default confidence since we don't have individual confidence scores
        }));
        setCandidateSkills((prev) => ({
          ...prev,
          [candidateId]: skills,
        }));
      }
    } catch (err) {
      console.error("Failed to load skills for candidate:", candidateId);
    }
  };

  const toggleCandidateDetails = (candidateId: string) => {
    if (selectedCandidate === candidateId) {
      setSelectedCandidate(null);
    } else {
      setSelectedCandidate(candidateId);
      loadCandidateSkills(candidateId);
    }
  };

  const handleDeleteCandidate = async (
    candidateId: string,
    candidateName: string
  ) => {
    if (
      !confirm(
        `Are you sure you want to delete ${candidateName}? This will permanently remove their data and CV file.`
      )
    ) {
      return;
    }

    setDeletingCandidates((prev) => new Set(prev).add(candidateId));

    try {
      const response = await apiService.deleteCandidate(candidateId);
      if (response.success) {
        // Remove from local state
        setCandidates((prev) =>
          prev.filter((c) => c.candidate_id !== candidateId)
        );
        // Clear from selected if it was selected
        if (selectedCandidate === candidateId) {
          setSelectedCandidate(null);
        }
        // Remove from skills cache
        setCandidateSkills((prev) => {
          const newSkills = { ...prev };
          delete newSkills[candidateId];
          return newSkills;
        });
        alert(`Success: ${response.message}`);
      }
    } catch (err: any) {
      // Handle 404 error (candidate already deleted)
      if (err.response?.status === 404) {
        alert(
          `Warning: Candidate ${candidateName} was already deleted or not found. Refreshing list...`
        );
        // Refresh the candidate list to get current data
        loadCandidates();
      } else {
        alert(
          `Error: Failed to delete candidate: ${err.message || "Unknown error"}`
        );
      }
    } finally {
      setDeletingCandidates((prev) => {
        const newSet = new Set(prev);
        newSet.delete(candidateId);
        return newSet;
      });
    }
  };

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-primary-600 animate-spin" />
          <span className="ml-2 text-gray-600">Loading candidates...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <p className="text-red-800">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          All Candidates
        </h1>
        <p className="text-lg text-gray-600">
          Browse all uploaded candidates and their extracted information
        </p>
      </div>

      <div className="mb-6 flex justify-between items-center">
        <p className="text-sm text-gray-600">
          Total: {candidates.length} candidates
        </p>
        <button
          onClick={loadCandidates}
          disabled={loading}
          className="flex items-center px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors disabled:opacity-50"
          title="Refresh candidates list"
        >
          <RefreshCw
            className={`w-4 h-4 mr-1 ${loading ? "animate-spin" : ""}`}
          />
          Refresh
        </button>
      </div>

      {candidates.length > 0 ? (
        <div className="space-y-4">
          {candidates.map((candidate) => (
            <div
              key={candidate.candidate_id}
              className="bg-white border border-gray-200 rounded-lg overflow-hidden"
            >
              <div
                className="p-6 cursor-pointer hover:bg-gray-50 transition-colors"
                onClick={() => toggleCandidateDetails(candidate.candidate_id)}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <div className="bg-primary-100 rounded-full p-3 mr-4">
                      <User className="w-6 h-6 text-primary-600" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">
                        {candidate.name}
                      </h3>
                      <p className="text-sm text-gray-600">
                        ID: {candidate.candidate_id}
                      </p>
                      {candidate.created_at && (
                        <div className="flex items-center mt-1">
                          <Calendar className="w-4 h-4 text-gray-400 mr-1" />
                          <span className="text-sm text-gray-500">
                            Added:{" "}
                            {new Date(
                              candidate.created_at
                            ).toLocaleDateString()}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteCandidate(
                          candidate.candidate_id,
                          candidate.name
                        );
                      }}
                      disabled={deletingCandidates.has(candidate.candidate_id)}
                      className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Delete candidate"
                    >
                      {deletingCandidates.has(candidate.candidate_id) ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </button>
                    <ChevronRight
                      className={`w-5 h-5 text-gray-400 transition-transform ${
                        selectedCandidate === candidate.candidate_id
                          ? "rotate-90"
                          : ""
                      }`}
                    />
                  </div>
                </div>

                {candidate.summary && (
                  <p className="mt-3 text-gray-700 line-clamp-2">
                    {candidate.summary}
                  </p>
                )}
              </div>

              {selectedCandidate === candidate.candidate_id && (
                <div className="border-t border-gray-200 bg-gray-50 p-6">
                  <div className="space-y-4">
                    {candidate.summary && (
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">
                          Summary
                        </h4>
                        <p className="text-gray-700">{candidate.summary}</p>
                      </div>
                    )}

                    {/* CV Download Link */}
                    {candidate.gcs_url && (
                      <div>
                        <h4 className="font-medium text-gray-900 mb-2">
                          CV Document
                        </h4>
                        <button
                          onClick={async () => {
                            try {
                              const blob = await apiService.downloadCV(
                                candidate.candidate_id
                              );
                              const url = window.URL.createObjectURL(blob);
                              const a = document.createElement("a");
                              a.href = url;
                              a.download = `${candidate.name}_CV.pdf`;
                              document.body.appendChild(a);
                              a.click();
                              window.URL.revokeObjectURL(url);
                              document.body.removeChild(a);
                            } catch (error) {
                              console.error("Error downloading CV:", error);
                              alert("Failed to download CV. Please try again.");
                            }
                          }}
                          className="inline-flex items-center text-primary-600 hover:text-primary-700 text-sm font-medium transition-colors"
                        >
                          <svg
                            className="w-4 h-4 mr-2"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                            />
                          </svg>
                          Download CV
                        </button>
                      </div>
                    )}

                    <div>
                      <h4 className="font-medium text-gray-900 mb-2 flex items-center">
                        <Tag className="w-4 h-4 mr-2" />
                        Extracted Skills
                      </h4>
                      {candidateSkills[candidate.candidate_id] ? (
                        candidateSkills[candidate.candidate_id].length > 0 ? (
                          <div className="flex flex-wrap gap-2">
                            {candidateSkills[candidate.candidate_id].map(
                              (skill) => (
                                <span
                                  key={skill.name}
                                  className="inline-flex items-center bg-primary-100 text-primary-800 px-3 py-1 rounded-full text-sm"
                                >
                                  {skill.name}
                                </span>
                              )
                            )}
                          </div>
                        ) : (
                          <p className="text-gray-500 text-sm">
                            No skills extracted
                          </p>
                        )
                      ) : (
                        <div className="flex items-center">
                          <Loader2 className="w-4 h-4 text-gray-400 animate-spin mr-2" />
                          <span className="text-sm text-gray-500">
                            Loading skills...
                          </span>
                        </div>
                      )}
                    </div>

                    <div className="text-xs text-gray-500 pt-2 border-t border-gray-200">
                      <p>Candidate ID: {candidate.candidate_id}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-white rounded-lg border border-gray-200">
          <User className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            No candidates found
          </h3>
          <p className="text-gray-600 mb-4">
            Upload some CVs to get started with candidate search
          </p>
          <a
            href="/upload"
            className="inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            Upload CV
          </a>
        </div>
      )}
    </div>
  );
}
