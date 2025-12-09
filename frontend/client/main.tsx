import "./global.css";
// Leaflet CSS will be dynamically imported in DelhiAirMapContent to avoid SSR issues

import { createRoot } from "react-dom/client";
import App from "./App";

const container = document.getElementById("root");
if (!container) {
  throw new Error("Root element not found");
}

createRoot(container).render(<App />);
