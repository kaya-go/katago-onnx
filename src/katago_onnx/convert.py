import pathlib

import onnx
import torch
from onnxconverter_common import float16
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process

from .utils import load_model


def convert_katago_torch_to_onnx(
    torch_model_path: str,
    onnx_model_path: str,
):
    """Convert a KataGo PyTorch checkpoint to ONNX format.

    Exports three versions:
    - FP32 (.fp32.onnx): Full precision, recommended for browser/WASM
    - FP16 (.fp16.onnx): Half precision, for native apps (CoreML, CUDA, WebGPU)
    - UINT8 (.uint8.onnx): Quantized, for memory-constrained devices

    Args:
        torch_model_path: Path to the PyTorch checkpoint.
        onnx_model_path: Base path for output ONNX models (suffixes will be added).
    """

    onnx_model_path_obj = pathlib.Path(onnx_model_path)

    # Load the PyTorch model
    model = load_model(torch_model_path, device="cpu")

    # Prepare inputs for ONNX export
    bin_input = torch.randn(1, 22, 19, 19, dtype=torch.float32)
    global_input = torch.randn(1, 19, dtype=torch.float32)
    model_inputs = (bin_input, global_input)

    # Define input and output names
    input_names = ["bin_input", "global_input"]
    output_names = [
        "policy",
        "value",
        "miscvalue",
        "moremiscvalue",
        "ownership",
        "scoring",
        "futurepos",
        "seki",
        "scorebelief",
    ]

    # Define dynamic axes (for dynamo=False)
    dynamic_axes = {
        "bin_input": {0: "batch_size", 2: "height", 3: "width"},
        "global_input": {0: "batch_size"},
        "policy": {0: "batch_size", 2: "moves"},
        "value": {0: "batch_size"},
        "miscvalue": {0: "batch_size"},
        "moremiscvalue": {0: "batch_size"},
        "ownership": {0: "batch_size", 2: "height", 3: "width"},
        "scoring": {0: "batch_size", 2: "height", 3: "width"},
        "futurepos": {0: "batch_size", 2: "height", 3: "width"},
        "seki": {0: "batch_size", 2: "height", 3: "width"},
        "scorebelief": {0: "batch_size"},
    }

    # Export the model to ONNX (FP32) - intermediate format
    # Note: We use dynamo=False because dynamo=True currently fails with
    # "No ONNX function found for aten.sym_size.int" for this model's dynamic shapes.
    model_fp32 = onnx_model_path_obj.with_suffix(".fp32.onnx")
    print(f"Exporting FP32 model to {model_fp32}...")
    torch.onnx.export(
        model,
        model_inputs,
        str(model_fp32),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )

    # Convert to FP16 (recommended for Apple Silicon and modern GPUs)
    # FP16 provides ~2x smaller size and faster inference on hardware with native FP16 support
    model_fp16 = onnx_model_path_obj.with_suffix(".fp16.onnx")
    print(f"Converting to FP16: {model_fp16}...")

    onnx_model = onnx.load(str(model_fp32))
    # Convert entire graph to FP16 including inputs/outputs
    # Note: Inference code must handle float16 I/O (e.g., using @petamoriken/float16 in JS)
    onnx_model_fp16 = float16.convert_float_to_float16(
        onnx_model,
        keep_io_types=False,  # Full FP16 graph including I/O
        min_positive_val=1e-7,
        max_finite_val=1e4,
    )
    onnx.save(onnx_model_fp16, model_fp16)
    print(f"FP16 model saved to: {model_fp16}")

    # Convert to UINT8 (for memory-constrained devices)
    model_prep = onnx_model_path_obj.with_suffix(".prep.onnx")
    model_uint8 = onnx_model_path_obj.with_suffix(".uint8.onnx")

    # Pre-process the model (shape inference and optimization)
    print("Pre-processing model for UINT8 quantization...")
    quant_pre_process(model_fp32, model_prep)

    # Quantize the model (UINT8) - ~4x smaller than FP32
    print(f"Quantizing model to {model_uint8}...")
    quantize_dynamic(
        model_input=model_prep,
        model_output=model_uint8,
        weight_type=QuantType.QUInt8,
    )
    print(f"UINT8 model saved to: {model_uint8}")

    # Clean up intermediate prep file only
    print("Cleaning up intermediate files...")
    model_prep.unlink()

    return str(onnx_model_path_obj)
