"""Text quality gate for README, code and commit messages.

Fails if it finds an em dash, an emoji, a TODO or TBD marker, or a banned word.
"""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BANNED_WORDS = ["Claude", "Stirling"]
MARKERS = ["TO" + "DO", "TB" + "D"]
EM_DASH = "—"
EMOJI = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
SKIP_DIRS = {".git", ".venv", "data", "__pycache__", ".pytest_cache"}
SUFFIXES = {".py", ".md", ".sh", ".txt", ".csv", ".json", ".toml", ".cfg", ".gitignore"}


def problems(text):
    found = []
    if EM_DASH in text:
        found.append("em dash")
    if EMOJI.search(text):
        found.append("emoji")
    for m in MARKERS:
        if re.search(rf"\b{m}\b", text):
            found.append(f"marker {m}")
    for w in BANNED_WORDS:
        if re.search(w, text, re.IGNORECASE):
            found.append(f"word {w}")
    return found


def main():
    failures = []
    for path in ROOT.rglob("*"):
        if path.is_dir() or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        if path.suffix not in SUFFIXES and path.name not in SUFFIXES:
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        for issue in problems(path.read_text(encoding="utf-8", errors="replace")):
            failures.append(f"{path.relative_to(ROOT)}: {issue}")
    try:
        log = subprocess.run(["git", "log", "--format=%B"], cwd=ROOT, capture_output=True, text=True,
                             encoding="utf-8", check=True).stdout
        for issue in problems(log):
            failures.append(f"commit messages: {issue}")
    except (OSError, subprocess.CalledProcessError):
        print("git log not available, commit messages not checked")
    for f in failures:
        print(f)
    print("check_text: " + ("FAILED" if failures else "passed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
