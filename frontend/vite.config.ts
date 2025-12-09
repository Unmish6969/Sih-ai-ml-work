import { defineConfig, Plugin } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { createServer } from "./server";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    host: "::",
    port: 8081,
    strictPort: true,
    fs: {
      // Allow Vite to serve the client, shared and repository root files
      // (index.html sits in the repo root). See:
      // https://vite.dev/config/server-options.html#server-fs-allow
      // Allow the current working directory explicitly (handles Windows absolute paths)
      // and existing project directories.
      allow: [process.cwd(), path.resolve(__dirname, "./client"), path.resolve(__dirname, "./shared")],
      deny: [".env", ".env.*", "*.{crt,pem}", "**/.git/**", "server/**"],
    },
  },
  build: {
    outDir: "dist/spa",
  },
  plugins: [react(), expressPlugin()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./client"),
      "@shared": path.resolve(__dirname, "./shared"),
    },
  },
}));

function expressPlugin(): Plugin {
  return {
    name: "express-plugin",
    apply: "serve", // Only apply during development (serve mode)
    configureServer(server) {
      const app = createServer();
      // Add Express app as middleware to Vite dev server
      server.middlewares.use(app);
    },
  };
}
