# PDF Parser Qwen

Backend for extracting scientific-paper abstracts and patent claim text with a local open model served through `llama.cpp`.

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

### One-command backend

This repo now includes a backend runner that:

1. Downloads a matching `llama.cpp` release for the current host platform.
2. Downloads the configured GGUF model.
3. Starts `llama-server`.
4. Starts the Flask backend.

Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_backend.py
```

By default, the runner reuses the previously installed `llama.cpp` release. To fetch the latest release explicitly:

```bash
python run_backend.py --update
```

The JSON backend will be available at `http://127.0.0.1:5001/api/parse`.

Multipart form fields:

- `pdf` required
- `task` optional: `abstract` or `first_claim`
- `custom_prompt` optional

Example:

```bash
curl -X POST http://127.0.0.1:5001/api/parse \
  -F "task=abstract" \
  -F "pdf=@/path/to/paper.pdf"
```

### Manual mode

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start `llama-server` in a separate shell. Example:

```bash
llama-server \
  -m /path/to/gemma-4-E2B-it-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 \
  --port 8080
```

Then start the Flask app:

```bash
python app.py
```

If your `llama-server` is running on another machine:

```bash
LLAMA_BASE_URL="http://192.168.1.51:8080" python app.py
```

The app now auto-detects the loaded model from `/v1/models`, so you usually do not need to set `LLAMA_MODEL` manually.

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
- `LLAMA_MODEL` default: unset; auto-detected from `/v1/models`
- `LLAMA_TIMEOUT_SECONDS` default: `120`
- `LLAMA_TEMPERATURE` default: `0.2`
- `LLAMA_ENABLE_THINKING` default: `1`
- `PDF_MAX_PAGES` default: `2`
- `PDF_MAX_CHARS` default: `9000`
- `LLAMA_MAX_TOKENS` default: `4800`
- `PATENT_MAX_TOKENS` default: `9600`
- `PAPER_USE_BLOCKS` default: `1`
- `BACKEND_HOST` default: `127.0.0.1`
- `BACKEND_PORT` default: `5001`
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

Backend runner configuration:

- `MODEL_REPO` default: `unsloth/gemma-4-E2B-it-GGUF`
- `MODEL_FILE` default: `gemma-4-E2B-it-UD-Q4_K_XL.gguf`
- `MODEL_URL` optional full override for model download
- `LLAMA_RELEASE_API` default: latest `ggml-org/llama.cpp` release API
- `LLAMA_SERVER_PATH` optional full override to an existing `llama-server` binary
- `LLAMA_SERVER_HOST` default: `127.0.0.1`
- `LLAMA_SERVER_PORT` default: `8080`
- `BACKEND_HOST` default: `127.0.0.1`
- `BACKEND_PORT` default: `5001`
- `LLAMA_N_GPU_LAYERS` default: `99`
- `LLAMA_SERVER_EXTRA_ARGS` optional extra args passed to `llama-server`
- `HF_TOKEN` optional Hugging Face token if you use a gated/private model
- `python run_backend.py --update` refreshes the pinned `llama.cpp` release
- By default the app sends `chat_template_kwargs.enable_thinking=true`. Set `LLAMA_ENABLE_THINKING=0` if you want to force thinking off.

## Notes

- For this task, text extraction is the intended path. The app does not use PDF page images or multimodal prompts.
- The request goes to `/v1/chat/completions`, so it should work with `llama-server`'s OpenAI-compatible API.
- Patent claims can be much longer than abstracts, so the patent task uses higher page, character, and token limits by default.
- If output quality is uneven, tune the page snippet sizes, patent window sizes, and task prompt before increasing the full-document fallback limit.
- Runtime downloads are stored under `downloads/`, `vendor/`, and `models/`, which are gitignored.
