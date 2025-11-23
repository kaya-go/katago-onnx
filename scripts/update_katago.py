import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/lightvector/KataGo.git"
# You can pin a specific commit or tag here for reproducibility
# BRANCH_OR_TAG = "v1.15.3"
BRANCH_OR_TAG = "v1.16.4"
SOURCE_SUBDIR = "python/katago"
DEST_DIR = Path(__file__).parent.parent / "src" / "katago"


def run_command(cmd, cwd=None):
    subprocess.check_call(cmd, shell=True, cwd=cwd)


def main():
    print(f"Updating KataGo library from {REPO_URL} ({BRANCH_OR_TAG})...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 1. Clone the repository
        print("Cloning repository...")
        run_command(
            f"git clone --depth 1 --branch {BRANCH_OR_TAG} {REPO_URL} repo",
            cwd=temp_dir,
        )

        repo_path = temp_path / "repo"
        source_path = repo_path / SOURCE_SUBDIR

        if not source_path.exists():
            raise FileNotFoundError(
                f"Could not find {SOURCE_SUBDIR} in the repository."
            )

        # 2. Clean destination
        if DEST_DIR.exists():
            print(f"Cleaning existing directory: {DEST_DIR}")
            shutil.rmtree(DEST_DIR)

        DEST_DIR.parent.mkdir(parents=True, exist_ok=True)

        # 3. Copy files
        print(f"Copying {source_path} to {DEST_DIR}...")
        shutil.copytree(source_path, DEST_DIR)

        # 4. Add metadata
        with open(DEST_DIR / "VENDOR_INFO.txt", "w") as f:
            f.write(f"Source: {REPO_URL}\n")
            f.write(f"Branch/Tag: {BRANCH_OR_TAG}\n")
            # Get commit hash
            commit_hash = (
                subprocess.check_output("git rev-parse HEAD", shell=True, cwd=repo_path)
                .decode()
                .strip()
            )
            f.write(f"Commit: {commit_hash}\n")

    print("Done!")


if __name__ == "__main__":
    main()
