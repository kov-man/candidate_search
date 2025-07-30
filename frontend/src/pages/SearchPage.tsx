import React, { useState } from "react";
import { Search, User, Star, FileText, Loader2 } from "lucide-react";
import { apiService } from "../services/api";
import type { SearchResponse, SearchQuery } from "../types";

export function SearchPage() {
  const [query, setQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResponse | null>(
    null
  );
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState("");
  const [thresholdEnabled, setThresholdEnabled] = useState(true);
  const [filteredResults, setFilteredResults] = useState<SearchResponse | null>(
    null
  );

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsSearching(true);
    setError("");

    try {
      const searchQuery: SearchQuery = {
        query: query.trim(),
        top_k: 10,
        similarity_threshold: 0.3,
      };

      const results = await apiService.searchCandidates(searchQuery);
      setSearchResults(results);
      setFilteredResults(results);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Search failed");
    } finally {
      setIsSearching(false);
    }
  };

  const handleThresholdToggle = () => {
    setThresholdEnabled(!thresholdEnabled);
  };

  const getFilteredResults = () => {
    if (!searchResults) return null;

    if (!thresholdEnabled) {
      return searchResults;
    }

    const filtered = {
      ...searchResults,
      results: searchResults.results.filter(
        (result) => result.similarity_score >= 0.5
      ),
    };

    return {
      ...filtered,
      total_results: filtered.results.length,
    };
  };

  // some example queries to help users
  const exampleQueries = [
    "Find developers with GCP and machine learning experience",
    "Who has worked with Kubernetes and Docker?",
    "Python developers with startup experience",
    "Data engineers with Spark and BigQuery",
    "Frontend developers with React and TypeScript",
  ];

  return (
    <div className="max-w-6xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Search Candidates
        </h1>
        <p className="text-lg text-gray-600">
          Use natural language to find the perfect candidates for your role
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="e.g., Find me candidates with React and Node.js experience"
            className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            disabled={isSearching}
          />
          <button
            type="submit"
            disabled={isSearching || !query.trim()}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-primary-600 text-white px-4 py-1.5 rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSearching ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              "Search"
            )}
          </button>
        </div>

        {/* threshold filter toggle */}
        <div className="mt-4 flex items-center justify-center">
          <div className="flex items-center space-x-3 bg-gray-50 rounded-lg px-4 py-2">
            <span className="text-sm font-medium text-gray-700">
              Show only results above 50% match
            </span>
            <button
              type="button"
              onClick={handleThresholdToggle}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${
                thresholdEnabled ? "bg-primary-600" : "bg-gray-200"
              }`}
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                  thresholdEnabled ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
            <span className="text-xs text-gray-500">
              {thresholdEnabled ? "Active" : "Inactive"}
            </span>
          </div>
        </div>
      </form>

      {/* example queries section */}
      <div className="mb-8">
        <p className="text-sm text-gray-600 mb-3">Try these examples:</p>
        <div className="flex flex-wrap gap-2">
          {exampleQueries.map((example, index) => (
            <button
              key={index}
              onClick={() => setQuery(example)}
              className="text-sm bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1 rounded-full transition-colors"
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <p className="text-red-800">{error}</p>
        </div>
      )}

      {/* search results */}
      {searchResults && (
        <div className="space-y-6">
          {/* AI generated answer */}
          {searchResults.rag_answer && (
            <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-6">
              <div className="flex items-center mb-3">
                <div className="bg-purple-100 rounded-full p-2 mr-3">
                  <svg
                    className="w-5 h-5 text-purple-600"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-purple-900">
                  AI Analysis
                </h3>
              </div>
              <div className="text-purple-800 leading-relaxed whitespace-pre-line">
                {searchResults.rag_answer}
              </div>
            </div>
          )}

          {/* Search summary */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-blue-900 mb-3">
              Search Results Summary
            </h3>
            <p className="text-blue-800 leading-relaxed">
              Found {searchResults.total_results} candidates using{" "}
              {searchResults.search_method || "semantic"} search. Results are
              ranked by relevance to your query.
            </p>
          </div>

          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-gray-900">
              Search Results ({getFilteredResults()?.total_results || 0} found)
              {thresholdEnabled &&
                searchResults.total_results !==
                  getFilteredResults()?.total_results && (
                  <span className="text-sm font-normal text-gray-500 ml-2">
                    (filtered from {searchResults.total_results})
                  </span>
                )}
            </h2>
            <p className="text-sm text-gray-600">
              Query: "{searchResults.query}"
            </p>
          </div>

          {/* results list */}
          {getFilteredResults()?.results &&
          getFilteredResults()!.results.length > 0 ? (
            <div className="space-y-4">
              {getFilteredResults()!.results.map((result, index) => (
                <div
                  key={result.candidate_id}
                  className="bg-white border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex items-center">
                      <div className="bg-primary-100 rounded-full p-2 mr-3">
                        <User className="w-5 h-5 text-primary-600" />
                      </div>
                      <div>
                        <h3 className="text-lg font-semibold text-gray-900">
                          {result.candidate_name}
                        </h3>
                      </div>
                    </div>
                    <div className="flex items-center text-sm text-gray-600">
                      <Star className="w-4 h-4 mr-1 text-yellow-500" />
                      {(result.similarity_score * 100).toFixed(1)}% match
                    </div>
                  </div>

                  {result.summary && (
                    <p className="text-gray-700 mb-4">{result.summary}</p>
                  )}

                  {result.skills.length > 0 && (
                    <div className="mb-4">
                      <h4 className="text-sm font-medium text-gray-900 mb-2">
                        Skills:
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {result.skills.map((skill, skillIndex) => (
                          <span
                            key={skillIndex}
                            className="inline-block bg-gray-100 text-gray-800 px-2 py-1 rounded text-sm"
                          >
                            {skill}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}

                  {result.relevant_text && (
                    <div className="bg-gray-50 rounded-lg p-3">
                      <div className="flex items-center mb-2">
                        <FileText className="w-4 h-4 text-gray-600 mr-2" />
                        <span className="text-sm font-medium text-gray-700">
                          Relevant CV excerpt:
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 italic">
                        "{result.relevant_text}"
                      </p>
                    </div>
                  )}

                  {/* Download CV button */}
                  {result.gcs_url && (
                    <div className="mt-4 flex items-center">
                      <button
                        onClick={async () => {
                          try {
                            const blob = await apiService.downloadCV(
                              result.candidate_id
                            );
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement("a");
                            a.href = url;
                            a.download = `${result.candidate_name}_CV.pdf`;
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
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center py-8">
              <Search className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">
                {thresholdEnabled && searchResults.results.length > 0
                  ? "No candidates meet the 50% threshold"
                  : "No candidates found"}
              </h3>
              <p className="text-gray-600">
                {thresholdEnabled && searchResults.results.length > 0
                  ? "Try disabling the threshold filter or adjusting your search query"
                  : "Try adjusting your search query or uploading more CVs"}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
