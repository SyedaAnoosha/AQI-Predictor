"""Environment loading shared by every entry point.

Looks for `.env` in the project root and in `src/`, so a pipeline finds the
same credentials whether it is launched from the repo root or from `src/`.
"""

import os
from dotenv import load_dotenv

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_env() -> None:
    """Load .env files. Values already in the environment always win."""
    for candidate in (
        os.path.join(_ROOT, ".env"),
        os.path.join(_ROOT, "src", ".env"),
    ):
        if os.path.exists(candidate):
            load_dotenv(dotenv_path=candidate, override=False)
