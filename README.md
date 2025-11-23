# KataGo ONNX Conversion Prototype

This repository is dedicated to exploring and prototyping the conversion of KataGo raw checkpoints (PyTorch models) to ONNX format. The ultimate goal is to enable these models to run in the browser via WebAssembly (WASM) for the [Kaya](https://github.com/kaya-go/kaya) project.

## Project Goal

The main objective is to create a pipeline that takes a KataGo PyTorch checkpoint and exports it to an ONNX model that is optimized for web inference. This involves:

1. Loading the PyTorch model.
2. Exporting it to ONNX.
3. Verifying the ONNX model against the original PyTorch model.
4. Optimizing the ONNX model for WASM execution (e.g., quantization).

## Prerequisites

This project uses [pixi](https://pixi.sh/) for dependency management. Please ensure you have `pixi` installed on your system.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/kaya-go/katago-onnx.git
   cd katago-onnx
   ```

2. Install dependencies:

   ```bash
   pixi install
   ```
