# Semantic Candidate Search - Frontend

React + TypeScript frontend for the Semantic Candidate Search application.

## Getting Started

### Prerequisites
- Node.js 18+ and npm
- Backend service running on port 8000

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at http://localhost:3000

### Build for Production

```bash
# Build optimized bundle
npm run build

# Preview production build
npm run preview
```

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **Lucide React** - Icons
- **React Dropzone** - File upload

## Project Structure

```
src/
├── components/          # Reusable components
│   └── Navbar.tsx      # Navigation bar
├── pages/              # Page components
│   ├── HomePage.tsx    # Landing page
│   ├── UploadPage.tsx  # CV upload page
│   ├── SearchPage.tsx  # Search interface
│   └── CandidatesPage.tsx # Candidates list
├── services/           # API services
│   └── api.ts         # Backend API client
├── types/             # TypeScript definitions
│   └── index.ts       # Type definitions
├── App.tsx            # Main application
├── main.tsx           # Entry point
└── index.css          # Global styles
```

## Styling

The application uses Tailwind CSS for styling with a custom color palette:

- Primary: Blue tones for branding
- Gray: Various shades for text and backgrounds
- Semantic colors: Green (success), Red (error), Blue (info)

## API Integration

The frontend communicates with the backend via REST API:

- Base URL: `/api` (proxied to `http://localhost:8000`)
- All API calls are handled through the `apiService` in `src/services/api.ts`

## Responsive Design

The application is fully responsive and works on:
- Desktop computers
- Tablets
- Mobile phones

## Deployment

### Using Vite Build
```bash
npm run build
npm run preview
```

### Using Static Hosting
Deploy the `dist` folder to any static hosting service like:
- Vercel
- Netlify
- AWS S3
- Firebase Hosting

## Features

- **File Upload**: Drag and drop CV files
- **Search Interface**: Natural language search with filters
- **Candidate Management**: View, edit, and delete candidates
- **Responsive Design**: Works on all device sizes
- **Real-time Updates**: Live search results
- **Download CVs**: Direct file downloads from cloud storage 