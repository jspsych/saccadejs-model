import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const root = dirname(fileURLToPath(import.meta.url));

// public/models -> ../../out/models and public/fixtures -> ../../fixtures are
// symlinks; `preserveSymlinks: false` (the default) makes vite serve through them.
export default defineConfig({
  root,
  publicDir: resolve(root, "public"),
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: {
    target: "es2022",
    rollupOptions: { input: { parity: resolve(root, "index.html") } },
  },
});
