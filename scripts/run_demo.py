from __future__ import annotations

import argparse
import subprocess
import sys

from email_thread_rag.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the dataset slice, ingest it, and start the API.")
    parser.add_argument("--skip-build", action="store_true", help="Skip dataset build + ingest and just run the API.")
    args = parser.parse_args()
    settings = get_settings()

    if not args.skip_build:
        subprocess.run([sys.executable, "-m", "email_thread_rag.scripts.build_dataset_slice"], check=True)
        subprocess.run([sys.executable, "-m", "email_thread_rag.scripts.ingest_corpus", "--build-slice"], check=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "email_thread_rag.app.main:app",
            "--host",
            settings.api_host,
            "--port",
            str(settings.api_port),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
