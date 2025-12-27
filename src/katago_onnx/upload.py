"""Upload ONNX models to Hugging Face Hub."""

import pathlib

from huggingface_hub import HfApi, create_repo

HF_REPO_ID = "kaya-go/kaya"


def upload_folder_to_huggingface(
    folder_path: str | pathlib.Path,
    repo_id: str = HF_REPO_ID,
    path_in_repo: str | None = None,
    commit_message: str | None = None,
) -> str:
    """Upload a folder to Hugging Face Hub, preserving the directory structure.

    Existing files in the target path are deleted before uploading.

    Args:
        folder_path: Path to the folder to upload.
        repo_id: The HF repository ID (e.g., "kaya-go/kaya").
        path_in_repo: Base path in the repo where folder contents will be stored.
                      If None, uploads to the root.
        commit_message: Custom commit message. If None, a default is used.

    Returns:
        The URL of the repository.
    """
    folder_path = pathlib.Path(folder_path)

    if path_in_repo is None:
        path_in_repo = ""

    if commit_message is None:
        commit_message = f"Upload {folder_path.name}"

    api = HfApi()

    # Create repo if it doesn't exist (will be a no-op if it exists)
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Upload the folder, deleting all existing files in the target path first
    url = api.upload_folder(
        folder_path=str(folder_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        delete_patterns="*",  # Delete all existing files in path_in_repo before upload
    )

    print(f"Uploaded folder {folder_path} to {url}")
    return url
