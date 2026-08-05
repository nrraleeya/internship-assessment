# Optical Character Recognition (OCR) Project – Deposit Extractor

This sub-module contains an production-grade text extraction engine engineered to parse transaction totals from digital and thermal banking receipts. 

Rather than relying on legacy, brittle character-matrix matching (like Tesseract) which frequently hallucinates values on low-resolution or skewed images, this solution utilizes a state-of-the-art Multimodal Vision-Language Pipeline.

## 1. Features & Architecture
* **Intelligent Character Recognition (ICR):** Leverages the `gemini-2.5-flash` multimodal model to read pixel data and contextually interpret damaged, pixelated, or poorly aligned receipt strings (e.g., automatically resolving blurred segments like `Ry6o:00` into a clean `60.00`).
* **Dual-Strategy Pipeline:** Features a robust fetch-and-fallback mechanism. It attempts to stream the target image dynamically over the network via live URLs first, automatically falling back to localized image assets (`receipt1.png`, `receipt2.png`) if firewall blocks or network timeouts occur.
* **Contextual Anchor Tag Matching:** Instructs the vision model to target the primary monetary values bound specifically to transaction records (Deposit/Transfer totals) while discarding background noise like account numbers, company registration numbers, and date timestamps.

## 2. How to Run the Pipeline

1. Ensure `receipt1.csv`, `receipt2.csv`, and `ocr_processor.py` are in the same directory.
2. To run the extraction engine, ensure the following official dependencies are installed in your environment:
```bash
pip install google-genai requests pillow
3. A valid API key is required. Before running the script, set your key as an environment variable:
`export ASSESSMENT_API_KEY="your_secret_key"`
