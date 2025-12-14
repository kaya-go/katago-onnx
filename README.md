# KataGo ONNX

Convert [KataGo](https://github.com/lightvector/KataGo) PyTorch checkpoints to ONNX format for web and cross-platform deployment.

The converted models are uploaded to Hugging Face: **[kaya-go/kaya](https://huggingface.co/kaya-go/kaya)**

These ONNX models power the [Kaya](https://github.com/kaya-go/kaya) app, a web-based Go application with AI-powered game analysis.

## Installation

```bash
pixi install
```

## Usage

### Convert models

Download KataGo checkpoints and convert them to ONNX:

```bash
pixi run katago-onnx convert ./artifacts/
```

### Upload to Hugging Face

Upload the converted models to Hugging Face Hub:

```bash
pixi run katago-onnx upload ./artifacts/ --repo-id kaya-go/kaya
```

## License

MIT
