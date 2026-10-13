"""Apply your decision on a proposal: approve, reject, or undo an approved change.

  python scripts/apply_proposal.py approve P-0003
  python scripts/apply_proposal.py reject  P-0003
  python scripts/apply_proposal.py undo    P-0003

Usually run by the "Proposal decision" workflow when you open an issue titled "[approve] P-0003".
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nbapred import learning, models, site  # noqa: E402


def main(argv):
    if len(argv) == 1:  # a raw issue title, e.g. "[approve] P-0003"
        m = re.match(r"^\s*\[(approve|reject|undo)\]\s*(P-\d{4})\s*$", argv[0], re.I)
        if not m:
            print("Title must look like: [approve] P-0003")
            return 2
        action, pid = m.group(1).lower(), m.group(2).upper()
    else:
        action, pid = argv[0].lower(), argv[1].upper()
    L = learning.load()
    msg = learning.apply(L, action, pid)
    learning.save(L)
    site._write("learning.json", learning.site_view(L, models.load_config()))
    print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
