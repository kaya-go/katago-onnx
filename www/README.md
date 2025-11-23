# KataGo ONNX Web Tester

This is a minimal frontend to test KataGo ONNX models in the browser.

## Usage

1.  **Serve the files**:
    It is recommended to run a local web server to avoid CORS or file access issues.

    ```bash
    cd www
    python -m http.server
    ```

    Then open [http://localhost:8000](http://localhost:8000) in your browser.

2.  **Load Model**:
    Drag and drop your `.onnx` model file into the "Load Model" area.

3.  **Load SGF**:
    Drag and drop an `.sgf` game file into the "Load SGF" area.

4.  **Run Inference**:
    Select a move number and click "Run Inference".
    The board will update to show the state at that move, and the results (Winrate, Score Lead, Top Moves) will be displayed.

## Limitations

- **Featurization**: The input features generated in JavaScript are a **simplified version** of the full KataGo features.
  - Ladders are not computed.
  - Area/Territory logic is not implemented.
  - Ko handling is basic (simple ko only).
  - This means the model's predictions might be slightly less accurate than the full Python version, especially in complex fighting situations involving ladders.
- **SGF Parsing**: The parser is basic and might not handle complex SGF properties or variations (it just follows the main line).
