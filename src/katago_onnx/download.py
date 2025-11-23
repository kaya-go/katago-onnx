import pathlib
import tempfile
import zipfile

import httpx
from tqdm.auto import tqdm


def download_and_extract_model(network_name: str, extract_to: str):
    torch_checkpoint_url = f"https://media.katagotraining.org/uploaded/networks/zips/kata1/{network_name}.zip"

    temp_dir = pathlib.Path(tempfile.gettempdir())
    zip_path = temp_dir / f"{network_name}.zip"

    with httpx.stream("GET", torch_checkpoint_url, follow_redirects=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))

        with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in response.iter_bytes():
                f.write(chunk)
                pbar.update(len(chunk))

    temp_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    # Move the model file
    model_path = temp_dir / "model.ckpt"

    # Move the model to the desired location
    extract_path = pathlib.Path(extract_to)
    extract_path.mkdir(parents=True, exist_ok=True)
    torch_model_path = extract_path / f"{network_name}.ckpt"
    model_path.rename(torch_model_path)

    # Clean up
    zip_path.unlink()

    return str(torch_model_path)
