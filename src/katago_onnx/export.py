import pathlib

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process

from .utils import load_model


def convert_katago_torch_to_onnx(torch_model_path: str, onnx_model_path: str):
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

    # Export the model to ONNX
    # Note: We use dynamo=False because dynamo=True currently fails with
    # "No ONNX function found for aten.sym_size.int" for this model's dynamic shapes.
    torch.onnx.export(
        model,
        model_inputs,
        onnx_model_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )

    # Define paths
    model_fp32 = onnx_model_path_obj
    model_prep = onnx_model_path_obj.with_suffix(".prep.onnx")
    model_quant = onnx_model_path_obj.with_suffix(".quant.onnx")

    # Pre-process the model (Shape inference and optimization)
    # This is recommended before quantization to ensure best results, especially for dynamic shapes
    print(f"Pre-processing model to {model_prep}...")
    quant_pre_process(model_fp32, model_prep)

    # Quantize the model (Int8)
    # This reduces the size by ~4x by converting weights from Float32 to Int8
    print(f"Quantizing model to {model_quant}...")
    quantize_dynamic(
        model_input=model_prep,
        model_output=model_quant,
        weight_type=QuantType.QUInt8,  # Quantize weights to Unsigned Int8
    )

    print(f"Quantized model saved to: {model_quant}")

    return str(model_quant)
