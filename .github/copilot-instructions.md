# KataGo ONNX Conversion Instructions

## Project Overview
This project prototypes the conversion of KataGo Go engine checkpoints (PyTorch) to ONNX format for WebAssembly (WASM) inference in the [Kaya](https://github.com/kaya-go/kaya) project.

## Environment & Dependencies
- **Manager**: `pixi` is the sole package manager. Do not use `pip` or `conda` directly.
- **Setup**: `pixi install`
- **Running**: `pixi run jupyter lab`
- **Adding Deps**: `pixi add <package>` (e.g., `pixi add onnx onnxruntime`)
- **Key Libs**: `pytorch`, `httpx` (for fetching models), `ruff` (linting).

## Architecture & Workflow
- **Current State**: Exploratory phase using Jupyter Notebooks in `notebooks/`.
- **Goal Pipeline**:
  1.  **Fetch/Load**: Download KataGo models (often hosted online) and load into PyTorch.
  2.  **Export**: Use `torch.onnx.export` with appropriate dynamic axes (batch size).
  3.  **Verify**: Compare PyTorch vs ONNX outputs with `numpy.testing`.
  4.  **Optimize**: Quantization for WASM (future step).

## Coding Conventions
- **Style**: Follow `ruff` defaults.
- **Notebooks**: Keep cells clean. Use markdown cells to document findings.
- **Paths**: Use relative paths from the project root or `notebooks/` directory carefully.

## Specific Patterns
- **Model Loading**: KataGo models usually have a specific architecture (ResNet-like). You may need to implement or import the model definition to load weights.
- **ONNX Export**:
  ```python
  torch.onnx.export(
      model,
      dummy_input,
      "model.onnx",
      input_names=["input"],
      output_names=["output"],
      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
  )
  ```
