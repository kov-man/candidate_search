import { Link, useLocation } from "react-router-dom";
import { Search, Upload, Users, Home } from "lucide-react";

export function Navbar() {
  const location = useLocation();

  const navItems = [
    { path: "/", label: "Home", icon: Home },
    { path: "/upload", label: "Upload CV", icon: Upload },
    { path: "/search", label: "Search", icon: Search },
    { path: "/candidates", label: "Candidates", icon: Users },
  ];

  return (
    <nav className="bg-white shadow-lg border-b border-gray-100 backdrop-blur-sm bg-white/95">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <Link
              to="/"
              className="text-xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent hover:from-secondary-600 hover:to-primary-600 transition-all duration-200"
            >
              Candidate Search
            </Link>
          </div>

          <div className="flex space-x-2">
            {navItems.map((item, index) => {
              const isActive = location.pathname === item.path;
              const Icon = item.icon;
              const isEven = index % 2 === 0;

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                    isActive
                      ? isEven
                        ? "bg-primary-100 text-primary-700 shadow-sm"
                        : "bg-secondary-100 text-secondary-700 shadow-sm"
                      : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
                  }`}
                >
                  <Icon className="w-4 h-4 mr-2" />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
