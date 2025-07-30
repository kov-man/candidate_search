import { Link } from "react-router-dom";
import { Upload, Search, Users, Brain, Database, Cloud } from "lucide-react";

export function HomePage() {
  const features = [
    {
      icon: Brain,
      title: "AI Search",
      description:
        "Use natural language to find candidates with semantic search powered by Vertex AI",
    },
    {
      icon: Upload,
      title: "Easy Upload",
      description:
        "Just upload PDF or DOCX files and we'll extract and analyze everything",
    },
    {
      icon: Database,
      title: "Knowledge Graph",
      description:
        "Build relationships between candidates and skills using Cloud Spanner",
    },
    {
      icon: Cloud,
      title: "GCP Native",
      description:
        "Built on Google Cloud Platform with Vertex AI, Storage, and Spanner",
    },
  ];

  return (
    <div className="max-w-7xl mx-auto">
      {/* Hero section */}
      <div className="text-center py-16">
        <h1 className="text-5xl font-bold text-gray-900 mb-6">
          Find the Right Candidates
          <span className="text-primary-600"> with AI</span>
        </h1>
        <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
          Upload resumes and search using natural language. Our AI understands
          context and finds the best matches for your requirements.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link
            to="/upload"
            className="bg-primary-600 text-white px-8 py-3 rounded-lg hover:bg-primary-700 transition-colors font-medium"
          >
            Upload CVs
          </Link>
          <Link
            to="/search"
            className="border border-primary-600 text-primary-600 px-8 py-3 rounded-lg hover:bg-primary-50 transition-colors font-medium"
          >
            Search Candidates
          </Link>
        </div>
      </div>

      {/* Features */}
      <div className="py-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            How it works
          </h2>
          <p className="text-lg text-gray-600">
            Powered by Google Cloud AI services for enterprise-grade performance
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          {features.map((feature, index) => (
            <div
              key={index}
              className="text-center p-6 rounded-lg hover:shadow-lg transition-shadow"
            >
              <div className="inline-flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mb-4">
                <feature.icon className="w-6 h-6 text-primary-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">
                {feature.title}
              </h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>

      {/* stats section */}
      <div className="bg-gray-50 rounded-2xl p-8 mb-16">
        <div className="grid md:grid-cols-3 gap-8 text-center">
          <div>
            <div className="text-3xl font-bold text-primary-600 mb-2">Fast</div>
            <p className="text-gray-600">Process 100+ CVs in minutes</p>
          </div>
          <div>
            <div className="text-3xl font-bold text-primary-600 mb-2">
              Smart
            </div>
            <p className="text-gray-600">AI-powered skill extraction</p>
          </div>
          <div>
            <div className="text-3xl font-bold text-primary-600 mb-2">
              Scalable
            </div>
            <p className="text-gray-600">Built for enterprise workloads</p>
          </div>
        </div>
      </div>

      {/* Example queries */}
      <div className="text-center py-16 bg-gradient-to-r from-primary-50 to-blue-50 rounded-2xl">
        <h2 className="text-3xl font-bold text-gray-900 mb-8">
          Try these example searches
        </h2>
        <div className="max-w-4xl mx-auto">
          <div className="grid md:grid-cols-2 gap-4 text-left">
            {[
              "Find React developers with startup experience",
              "Who has worked with GCP and machine learning?",
              "Python engineers with 5+ years experience",
              "Data scientists with PhD in computer science",
              "Full-stack developers familiar with microservices",
              "DevOps engineers with Kubernetes experience",
            ].map((query, index) => (
              <div
                key={index}
                className="bg-white p-4 rounded-lg shadow-sm border"
              >
                <div className="flex items-center">
                  <Search className="w-4 h-4 text-gray-400 mr-3" />
                  <span className="text-gray-700">"{query}"</span>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="mt-8">
          <Link
            to="/search"
            className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors"
          >
            Try it now
          </Link>
        </div>
      </div>

      {/* Get started */}
      <div className="text-center py-16">
        <h2 className="text-3xl font-bold text-gray-900 mb-4">
          Ready to get started?
        </h2>
        <p className="text-lg text-gray-600 mb-8">
          Upload your first CV and see the magic happen
        </p>
        <Link
          to="/upload"
          className="bg-primary-600 text-white px-8 py-3 rounded-lg hover:bg-primary-700 transition-colors font-medium inline-flex items-center"
        >
          <Upload className="w-5 h-5 mr-2" />
          Upload CVs
        </Link>
      </div>
    </div>
  );
}
