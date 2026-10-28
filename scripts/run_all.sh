#!/usr/bin/env bash
# Run the tests, the smoke pipeline and (unless --smoke is given) the full pipeline,
# then render the README tables and run the text check.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

"$PY" -m pytest -q tests
"$PY" scripts/run_pipeline.py --smoke
# Render the README tables from the smoke results into a copy, to check the rendering path.
cp README.md results/smoke/README.md
"$PY" scripts/render_readme_tables.py --results results/smoke --readme results/smoke/README.md

if [[ "${1:-}" != "--smoke" ]]; then
    "$PY" scripts/run_pipeline.py
    "$PY" scripts/render_readme_tables.py
fi
"$PY" scripts/check_text.py
