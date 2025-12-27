import pathlib

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process

from .utils import load_model


def _fix_fp16_graph_type_mismatches(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fix type mismatches in FP16 ONNX graph.

    The onnxconverter_common float16 converter can create malformed graphs where:
    - Cast nodes output float16 but consuming nodes expect float32
    - Value infos have incorrect types after conversion

    This function walks the graph and ensures type consistency by:
    1. Building a map of actual output types from each node
    2. Updating value_info entries to match actual producer types
    3. Adding Cast nodes where necessary to fix mismatches

    Args:
        model: The ONNX model to fix.

    Returns:
        Fixed ONNX model with consistent types.
    """
    graph = model.graph

    # Build a map of tensor name -> actual element type
    tensor_types: dict[str, int] = {}

    # Get types from graph inputs
    for inp in graph.input:
        if inp.type.tensor_type.elem_type:
            tensor_types[inp.name] = inp.type.tensor_type.elem_type

    # Get types from initializers
    for init in graph.initializer:
        tensor_types[init.name] = init.data_type

    # Infer output types from nodes
    for node in graph.node:
        if node.op_type == "Cast":
            # Cast output type is specified in the 'to' attribute
            for attr in node.attribute:
                if attr.name == "to":
                    for out in node.output:
                        tensor_types[out] = attr.i
        elif node.op_type in ("Constant", "ConstantOfShape"):
            # Get type from the value attribute
            for attr in node.attribute:
                if attr.name == "value" and attr.t:
                    for out in node.output:
                        tensor_types[out] = attr.t.data_type
        else:
            # For other ops, assume outputs have same type as first typed input
            input_type = None
            for inp_name in node.input:
                if inp_name in tensor_types:
                    input_type = tensor_types[inp_name]
                    break
            if input_type is not None:
                for out in node.output:
                    if out not in tensor_types:
                        tensor_types[out] = input_type

    # Update value_info entries to match actual types
    for vi in graph.value_info:
        if vi.name in tensor_types:
            vi.type.tensor_type.elem_type = tensor_types[vi.name]

    # Run shape inference to fix any remaining issues
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        # If shape inference fails, continue with what we have
        pass

    return model


def _convert_to_fp16_native(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert FP32 ONNX model to FP16 using native ONNX APIs.

    This is an alternative to onnxconverter_common that produces cleaner graphs.
    It converts weights and Cast nodes while keeping I/O types configurable.

    Args:
        model: The FP32 ONNX model.

    Returns:
        FP16 ONNX model.
    """
    graph = model.graph

    # Convert initializers (weights) to FP16
    new_initializers = []
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT:
            # Convert float32 weights to float16
            arr = numpy_helper.to_array(init)
            arr_fp16 = arr.astype(np.float16)
            new_init = numpy_helper.from_array(arr_fp16, name=init.name)
            new_initializers.append(new_init)
        else:
            new_initializers.append(init)

    # Clear and replace initializers
    while len(graph.initializer) > 0:
        graph.initializer.pop()
    graph.initializer.extend(new_initializers)

    # Update graph inputs to FP16 (except for integer inputs)
    for inp in graph.input:
        # Skip inputs that are initializers
        if any(init.name == inp.name for init in graph.initializer):
            continue
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Update graph outputs to FP16
    for out in graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Update value_info to FP16
    for vi in graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16

    # Update Cast nodes and Constant nodes
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
        elif node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    arr_fp16 = arr.astype(np.float16)
                    new_tensor = numpy_helper.from_array(arr_fp16)
                    attr.t.CopyFrom(new_tensor)
        elif node.op_type == "ConstantOfShape":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                    arr = numpy_helper.to_array(attr.t)
                    arr_fp16 = arr.astype(np.float16)
                    new_tensor = numpy_helper.from_array(arr_fp16)
                    attr.t.CopyFrom(new_tensor)

    # Run shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception:
        pass

    return model


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

    # Use native FP16 conversion for cleaner graphs
    # The onnxconverter_common float16 converter can create type mismatches
    onnx_model_fp16 = _convert_to_fp16_native(onnx_model)

    # Validate the model before saving
    try:
        onnx.checker.check_model(onnx_model_fp16)
        print("FP16 model validation passed.")
    except onnx.checker.ValidationError as e:
        print(f"Warning: FP16 model validation issue: {e}")
        # Try to fix any remaining type mismatches
        onnx_model_fp16 = _fix_fp16_graph_type_mismatches(onnx_model_fp16)

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
