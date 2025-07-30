import React, { useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, FileText, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { apiService } from "../services/api";
import type { UploadResponse } from "../types";

export function UploadPage() {
  const [uploadStatus, setUploadStatus] = useState<
    "idle" | "uploading" | "success" | "error"
  >("idle");
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>("");

  const onDrop = async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    const file = acceptedFiles[0];
    setUploadStatus("uploading");
    setErrorMessage("");

    try {
      const result = await apiService.uploadCV(file);
      setUploadResult(result);
      setUploadStatus("success");
    } catch (error: any) {
      setErrorMessage(error.response?.data?.detail || "Failed to upload file");
      setUploadStatus("error");
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
    },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const resetUpload = () => {
    setUploadStatus("idle");
    setUploadResult(null);
    setErrorMessage("");
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">Upload CV</h1>
        <p className="text-lg text-gray-600">
          Upload candidate CVs in PDF or DOCX format for AI-powered analysis
        </p>
      </div>

      {uploadStatus === "idle" && (
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
            isDragActive
              ? "border-primary-400 bg-primary-50"
              : "border-gray-300 hover:border-gray-400"
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="w-16 h-16 text-gray-400 mx-auto mb-4" />

          {isDragActive ? (
            <p className="text-lg text-primary-600">Drop the CV here...</p>
          ) : (
            <>
              <p className="text-lg text-gray-600 mb-2">
                Drag and drop a CV here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Supports PDF and DOCX files up to 10MB
              </p>
            </>
          )}
        </div>
      )}

      {uploadStatus === "uploading" && (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
          <Loader2 className="w-16 h-16 text-primary-600 mx-auto mb-4 animate-spin" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">
            Processing CV...
          </h3>
          <p className="text-gray-600">
            Extracting text, identifying skills, and generating embeddings
          </p>
        </div>
      )}

      {uploadStatus === "success" && uploadResult && (
        <div className="bg-green-50 rounded-lg border border-green-200 p-8">
          <div className="flex items-center mb-4">
            <CheckCircle className="w-8 h-8 text-green-600 mr-3" />
            <h3 className="text-lg font-semibold text-green-900">
              Upload Successful!
            </h3>
          </div>

          <div className="mb-6">
            <p className="text-green-800 mb-2">{uploadResult.message}</p>
            <div className="text-sm text-green-700">
              <p>
                <strong>Candidate ID:</strong> {uploadResult.candidate_id}
              </p>
              <p>
                <strong>Filename:</strong> {uploadResult.filename}
              </p>
            </div>
          </div>

          {uploadResult.skills_extracted.length > 0 && (
            <div className="mb-6">
              <h4 className="font-semibold text-green-900 mb-2">
                Extracted Skills:
              </h4>
              <div className="flex flex-wrap gap-2">
                {uploadResult.skills_extracted.map((skill, index) => (
                  <span
                    key={index}
                    className="inline-block bg-green-100 text-green-800 px-3 py-1 rounded-full text-sm"
                  >
                    {skill}
                  </span>
                ))}
              </div>
            </div>
          )}

          <button
            onClick={resetUpload}
            className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors"
          >
            Upload Another CV
          </button>
        </div>
      )}

      {uploadStatus === "error" && (
        <div className="bg-red-50 rounded-lg border border-red-200 p-8">
          <div className="flex items-center mb-4">
            <XCircle className="w-8 h-8 text-red-600 mr-3" />
            <h3 className="text-lg font-semibold text-red-900">
              Upload Failed
            </h3>
          </div>

          <p className="text-red-800 mb-6">{errorMessage}</p>

          <button
            onClick={resetUpload}
            className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors"
          >
            Try Again
          </button>
        </div>
      )}

      {/* Instructions */}
      <div className="mt-12 bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          How it works:
        </h3>
        <div className="grid md:grid-cols-3 gap-6">
          <div className="flex items-start">
            <div className="bg-primary-100 rounded-full p-2 mr-3 mt-1">
              <FileText className="w-4 h-4 text-primary-600" />
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-1">
                Text Extraction
              </h4>
              <p className="text-sm text-gray-600">
                AI extracts and processes all text content from your CV
              </p>
            </div>
          </div>

          <div className="flex items-start">
            <div className="bg-primary-100 rounded-full p-2 mr-3 mt-1">
              <Upload className="w-4 h-4 text-primary-600" />
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-1">
                Skill Detection
              </h4>
              <p className="text-sm text-gray-600">
                Advanced NLP identifies skills, technologies, and expertise
              </p>
            </div>
          </div>

          <div className="flex items-start">
            <div className="bg-primary-100 rounded-full p-2 mr-3 mt-1">
              <CheckCircle className="w-4 h-4 text-primary-600" />
            </div>
            <div>
              <h4 className="font-medium text-gray-900 mb-1">
                Ready to Search
              </h4>
              <p className="text-sm text-gray-600">
                Candidate becomes searchable with semantic AI matching
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
