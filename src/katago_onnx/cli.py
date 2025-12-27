import pathlib
import shutil
from typing import Annotated

import typer
from tqdm.auto import tqdm

from .convert import convert_katago_torch_to_onnx
from .download import download_and_extract_model
from .upload import HF_REPO_ID, upload_folder_to_huggingface

app = typer.Typer()

# Path to the HF model card relative to the package
HF_MODEL_CARD_PATH = pathlib.Path(__file__).parent.parent.parent / "HF_MODEL_CARD.md"

NETWORK_NAMES = ["kata1-b28c512nbt-adam-s11165M-d5387M", "kata1-b28c512nbt-s12043015936-d5616446734"]


@app.command()
def convert(
    base_dir: Annotated[str, typer.Argument(help="Directory to save the converted models")],
    network_names: Annotated[
        list[str], typer.Option("--networks", "-n", help="List of network names to convert")
    ] = NETWORK_NAMES,
):
    """Download KataGo models and convert them to ONNX format."""
    base_path = pathlib.Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    for network_name in tqdm(network_names, desc="Converting models"):
        # Download and extract the model
        torch_model_path = download_and_extract_model(network_name, base_dir)

        # Define ONNX model path
        onnx_model_path = base_path / network_name / f"{network_name}.onnx"
        onnx_model_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to ONNX
        convert_katago_torch_to_onnx(
            torch_model_path=torch_model_path,
            onnx_model_path=str(onnx_model_path),
        )

        # Cleanup original torch model file
        pathlib.Path(torch_model_path).unlink()


@app.command()
def upload(
    folder_path: Annotated[str, typer.Argument(help="Path to the folder to upload")],
    repo_id: Annotated[str, typer.Option("--repo-id", "-r", help="HF repository ID")] = HF_REPO_ID,
    commit_message: Annotated[str | None, typer.Option("--message", "-m", help="Commit message")] = None,
):
    """Upload a folder to Hugging Face Hub, preserving directory structure."""
    folder = pathlib.Path(folder_path)

    # Copy HF_MODEL_CARD.md to README.md in the upload folder
    if not HF_MODEL_CARD_PATH.exists():
        raise FileNotFoundError(f"Model card not found: {HF_MODEL_CARD_PATH}")

    readme_dest = folder / "README.md"
    shutil.copy(HF_MODEL_CARD_PATH, readme_dest)
    print(f"Copied {HF_MODEL_CARD_PATH.name} to {readme_dest}")

    upload_folder_to_huggingface(
        folder_path=folder_path,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    print("Upload complete!")


if __name__ == "__main__":
    app()
