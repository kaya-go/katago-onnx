import pathlib

import typer
from tqdm.auto import tqdm

from .download import download_and_extract_model
from .export import convert_katago_torch_to_onnx

app = typer.Typer()


NETWORK_NAMES = ["kata1-b28c512nbt-adam-s11165M-d5387M"]


@app.command()
def main(base_dir: str):
    for network_name in tqdm(NETWORK_NAMES):
        # Download and extract the model
        torch_model_path = download_and_extract_model(network_name, base_dir)

        # Define ONNX model path
        onnx_model_path = pathlib.Path(base_dir) / f"{network_name}.onnx"

        # Convert to ONNX
        convert_katago_torch_to_onnx(
            torch_model_path=torch_model_path,
            onnx_model_path=str(onnx_model_path),
        )

        # Cleanup torch model file and prep.onnx file
        pathlib.Path(torch_model_path).unlink()
        prep_onnx_path = pathlib.Path(base_dir) / f"{network_name}.prep.onnx"
        prep_onnx_path.unlink()


if __name__ == "__main__":
    app()
