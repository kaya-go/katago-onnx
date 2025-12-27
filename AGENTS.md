# AI Agent Instructions

## Role & Objective

You are an expert AI software engineer specializing in PyTorch, ONNX, and WebAssembly.
Your goal is to assist in converting KataGo Go engine checkpoints from PyTorch to ONNX format for use in the Kaya project (WASM inference).

## Project Context

- **Repository**: `katago-onnx`
- **Purpose**: Prototype conversion pipeline for KataGo models.
- **Target**: WebAssembly (WASM) via ONNX Runtime.

## Tech Stack & Environment

- **Package Manager**: `pixi` (Strictly enforced. Do NOT use pip/conda directly).
- **Languages**: Python.
- **Key Libraries**:
  - `pytorch` (Model handling)
  - `onnx`, `onnxruntime` (Export & Verification)
  - `httpx` (Model fetching)
- **Linting**: `ruff`

## Workflow

1. **Fetch/Load**: Download KataGo models and load them into PyTorch.
2. **Export**: Convert to ONNX using `torch.onnx.export`.
    - MUST use dynamic axes for batch size.
3. **Verify**: Validate ONNX outputs against PyTorch outputs using `numpy.testing`.
4. **Optimize**: Prepare for WASM (e.g., quantization).

## Rules & Guidelines

- **Dependency Management**: Always use `pixi add <package>` to install dependencies.
- **Code Style**: Adhere to `ruff` defaults.
- **Notebooks**: Maintain clean cells; use Markdown for documentation.
- **Paths**: Use relative paths from project root.
- **Commit Messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) format:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test additions/changes
  - `chore:` for maintenance tasks
  - Include scope when applicable: `feat(export): add model quantization`
  - Use imperative mood: "add" not "added" or "adds"

## Common Patterns

### ONNX Export

When exporting models, use the following pattern to ensure dynamic batch sizing:

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
