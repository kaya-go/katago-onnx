---
license: mit
tags:
  - onnx
  - game-ai
  - go
  - baduk
  - weiqi
  - katago
library_name: onnxruntime
pipeline_tag: other
---

# Kaya - KataGo ONNX Models

This repository contains ONNX-converted versions of [KataGo](https://github.com/lightvector/KataGo) neural network models for the game of Go (Baduk/Weiqi).

These models power the [Kaya](https://github.com/kaya-go/kaya) app, a web-based Go application with AI-powered game analysis and move suggestions.

## Model Description

These models are converted from the official KataGo PyTorch checkpoints to ONNX format for use in web-based and cross-platform applications.

### Available Models

| Model                                       | Description             |
| ------------------------------------------- | ----------------------- |
| `kata1-b28c512nbt-adam-s11165M-d5387M`      | 28 blocks, 512 channels |
| `kata1-b28c512nbt-s12043015936-d5616446734` | 28 blocks, 512 channels |

Each model is available in three versions:

- **`.fp32.onnx`** - Full precision (FP32) - Recommended for browser/WASM
- **`.fp16.onnx`** - Half precision (FP16) - For native apps (CoreML, CUDA, WebGPU)
- **`.uint8.onnx`** - Quantized (UINT8) - ~4x smaller, for memory-constrained devices

## Usage

### With ONNX Runtime (Python)

```python
import onnxruntime as ort
import numpy as np

# Load the model (use .fp32.onnx for browser/WASM, .fp16.onnx for native apps)
session = ort.InferenceSession("kata1-b28c512nbt-adam-s11165M-d5387M.fp32.onnx")

# Prepare inputs (batch_size, channels, height, width)
bin_input = np.random.randn(1, 22, 19, 19).astype(np.float32)
global_input = np.random.randn(1, 19).astype(np.float32)

# Run inference
outputs = session.run(None, {
    "bin_input": bin_input,
    "global_input": global_input
})

policy, value, miscvalue, moremiscvalue, ownership, scoring, futurepos, seki, scorebelief = outputs
```

### With ONNX Runtime Web (JavaScript)

```javascript
import * as ort from "onnxruntime-web";

// Use .fp32.onnx for WASM backend, or .uint8.onnx for smaller download size
const session = await ort.InferenceSession.create(
  "kata1-b28c512nbt-adam-s11165M-d5387M.fp32.onnx"
);

const binInput = new ort.Tensor(
  "float32",
  new Float32Array(1 * 22 * 19 * 19),
  [1, 22, 19, 19]
);
const globalInput = new ort.Tensor(
  "float32",
  new Float32Array(1 * 19),
  [1, 19]
);

const results = await session.run({
  bin_input: binInput,
  global_input: globalInput,
});
```

## Model Inputs

| Name           | Shape                        | Description                    |
| -------------- | ---------------------------- | ------------------------------ |
| `bin_input`    | `[batch, 22, height, width]` | Board features (binary planes) |
| `global_input` | `[batch, 19]`                | Global features                |

## Model Outputs

| Name            | Shape                       | Description                    |
| --------------- | --------------------------- | ------------------------------ |
| `policy`        | `[batch, 2, moves]`         | Move policy logits             |
| `value`         | `[batch, 3]`                | Win/loss/draw predictions      |
| `miscvalue`     | `[batch, ...]`              | Miscellaneous value outputs    |
| `moremiscvalue` | `[batch, ...]`              | Additional value outputs       |
| `ownership`     | `[batch, 1, height, width]` | Territory ownership prediction |
| `scoring`       | `[batch, 1, height, width]` | Scoring prediction             |
| `futurepos`     | `[batch, 2, height, width]` | Future position prediction     |
| `seki`          | `[batch, 4, height, width]` | Seki detection                 |
| `scorebelief`   | `[batch, ...]`              | Score belief distribution      |

## Original Source

These models are derived from the [KataGo](https://github.com/lightvector/KataGo) project by David J. Wu (lightvector).

- **Original Repository**: https://github.com/lightvector/KataGo
- **Training Data**: https://katagotraining.org/
- **Original Checkpoints**: https://media.katagotraining.org/

## License

The original KataGo neural network weights are released under the [MIT License](https://github.com/lightvector/KataGo/blob/master/LICENSE).

This ONNX conversion and the associated tooling are also released under the MIT License.

## Citation

If you use these models, please cite the original KataGo paper:

```bibtex
@article{wu2019accelerating,
  title={Accelerating Self-Play Learning in Go},
  author={Wu, David J.},
  journal={arXiv preprint arXiv:1902.10565},
  year={2019}
}
```

## Conversion Details

- **Conversion Tool**: [katago-onnx](https://github.com/kaya-go/katago-onnx)
- **ONNX Opset**: 17
- **FP16 Conversion**: Internal computations in FP16, I/O remains FP32 for compatibility
- **UINT8 Quantization**: Dynamic quantization with QUInt8 weights
- **Dynamic Axes**: Batch size, board height/width are dynamic

## Acknowledgments

Special thanks to:

- David J. Wu (lightvector) for creating KataGo
- The KataGo training community for providing trained networks
