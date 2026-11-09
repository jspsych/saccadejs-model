import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root,
  build: {
    target: "es2022",
    rollupOptions: { input: { timing: resolve(root, "index.html") } },
  },
});
