// Wire the page's public/ dir to the export outputs.
//
// The parity page serves two directories it does not own: the exported model
// (../out/models, gitignored — it only exists after export_onnx.py runs) and the
// tracked parity fixtures (../fixtures). Both are reached through symlinks in
// public/. Those symlinks are NOT committed: a symlink into gitignored territory
// dangles in a fresh clone, and vite's prepareOutDir does a statSync over
// publicDir and dies on it. So create them here, and create the model directory
// too, so a build works before anything has been exported. An empty models/ just
// means the page 404s at runtime until you run the export, which is the honest
// behaviour.
import { mkdirSync, symlinkSync, statSync, lstatSync, unlinkSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const links = [
  ["public/models", "../../out/models"],
  ["public/fixtures", "../../fixtures"],
];

mkdirSync(join(root, "../out/models"), { recursive: true });
mkdirSync(join(root, "public"), { recursive: true });

for (const [linkPath, target] of links) {
  const abs = join(root, linkPath);
  try {
    lstatSync(abs);
    try {
      statSync(abs); // resolves? then it is a good link (or a real dir)
      continue;
    } catch {
      unlinkSync(abs); // dangling — replace it
    }
  } catch {
    /* absent */
  }
  symlinkSync(target, abs, "dir");
  console.log(`linked ${linkPath} -> ${target}`);
}
