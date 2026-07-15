import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

// Apply the saved theme before first paint to avoid a flash of the wrong theme.
(() => {
  try {
    const saved = localStorage.getItem("theme");
    document.documentElement.setAttribute("data-theme", saved === "light" ? "light" : "dark");
  } catch {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
