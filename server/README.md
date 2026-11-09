# Server scripts

These run on the lab server (`cogsciresearch.vassar.edu`, an AWS EC2 instance)
under `/var/www/html/jdeleeuw/<dataset>/data/`. Retrieved from the server and
committed here for the record. The hardcoded OSF token has been replaced with an
`OSF_TOKEN` environment-variable read in every `sendosf_private.sh`.

Layout mirrors the server, one folder per collection:

- `eyedata23/` — main dataset
  - `extractjpg.sh` — one JPG frame per `.webm` (basic ffmpeg).
  - `sendosf_private.sh` — tar `json`/`jpg`, split jpg into 1GB parts, upload to
    the **private** OSF (json `vmdu6`, jpg `tskx2`/`ksfjc`).
- `eyedata24/` — test dataset (288 trials/person)
  - `extractjpg.sh` — multi-method ffmpeg with fallbacks + logging.
  - `extractjpgnew.sh` — raw-YUV / zscale color-space variant.
  - `sendosf_private.sh` — same upload flow (json `c5ux8`, jpg `p3xyj`/`#####`).

## Usage

```bash
export OSF_TOKEN=...   # write-access personal access token
cd /var/www/html/jdeleeuw/eyedata23/data
./extractjpg.sh
./sendosf_private.sh
```

NOTE: OSF won't overwrite, so delete the existing tars in the target OSF project
before re-running `sendosf_private.sh`. (`eyedata24/sendosf_private.sh` has an
incomplete second jpg upload URL `#####` on the server -- fill it in before use.)
