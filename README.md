# PDF Parser Qwen

Rough v1 for extracting the abstract from a scientific PDF with a local Qwen model served through `llama.cpp`.

## What it does

1. Accepts a PDF upload in a tiny web UI.
2. Extracts text from the PDF with PyMuPDF.
3. Sends that text to a local `llama-server` instance.
4. Returns only the requested section, such as the abstract or first patent claim.

This version assumes:

- PDFs are born-digital, not scanned.
- A local `llama-server` is already running.
- The model is good enough at recovering the target span from noisy extracted text.

## Quick start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start `llama-server` in a separate shell. Example:

```bash
llama-server \
  -m /path/to/Qwen3.5-4B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080
```

If your `llama-server` requires a different model name in the OpenAI-compatible request, set:

```bash
export LLAMA_MODEL=qwen3.5-4b
```

Run the app:

```bash
flask --app app run --debug
```

Open `http://127.0.0.1:5000`.

## Prompt control

The UI now supports:

- A task selector for `Paper Abstract` and `Patent First Claim`
- A `Prompt Override` textarea if you want to swap in a completely custom system prompt

This app still uses text extraction only. It does not render PDF pages to images or send visual inputs to the model.

## Patent flow

The patent task no longer assumes the target text is near the front of the document.

It now:

1. Extracts text from every page.
2. Builds page summaries.
3. Asks the model to rank the pages most likely to contain the start of claim 1.
4. Extracts from a focused page window around the best candidate.
5. Expands that window if the result looks too weak.
6. Optionally falls back to the full document if the text stays within the configured limit.

## Config

The app reads a few environment variables:

- `LLAMA_BASE_URL` default: `http://127.0.0.1:8080`
- `LLAMA_MODEL` default: `qwen3.5-4b`
- `LLAMA_TIMEOUT_SECONDS` default: `120`
- `PDF_MAX_PAGES` default: `4`
- `PDF_MAX_CHARS` default: `18000`
- `LLAMA_MAX_TOKENS` default: `1200`
- `PATENT_MAX_TOKENS` default: `2200`
- `PAGE_SNIPPET_HEAD_CHARS` default: `1400`
- `PAGE_SNIPPET_TAIL_CHARS` default: `320`
- `PATENT_WINDOW_BACK` default: `1`
- `PATENT_WINDOW_FORWARD` default: `2`
- `PATENT_EXPANSION_STEP` default: `2`
- `PATENT_EXPANSION_ATTEMPTS` default: `3`
- `PATENT_MIN_RESULT_CHARS` default: `160`
- `PATENT_MAX_CANDIDATES` default: `6`
- `PATENT_ALLOW_FULL_DOC_FALLBACK` default: `1`
- `PATENT_FULL_DOC_MAX_CHARS` default: `120000`

## Notes

- For this task, text extraction is the intended path. The app does not use PDF page images or multimodal prompts.
- The request goes to `/v1/chat/completions`, so it should work with `llama-server`'s OpenAI-compatible API.
- Patent claims can be much longer than abstracts, so the patent task uses higher page, character, and token limits by default.
- If output quality is uneven, tune the page snippet sizes, patent window sizes, and task prompt before increasing the full-document fallback limit.
