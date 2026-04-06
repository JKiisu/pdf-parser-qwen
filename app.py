import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional

import fitz
import requests
from flask import Flask, jsonify, redirect, render_template, request, url_for


LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "").strip()
LLAMA_TIMEOUT = int(os.getenv("LLAMA_TIMEOUT_SECONDS", "120"))
LLAMA_ENABLE_THINKING = os.getenv("LLAMA_ENABLE_THINKING", "0") == "1"
LLAMA_DEBUG = os.getenv("LLAMA_DEBUG", "0") == "1"
PAPER_USE_BLOCKS = os.getenv("PAPER_USE_BLOCKS", "1") == "1"
BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5001"))
DEFAULT_MAX_PAGES = int(os.getenv("PDF_MAX_PAGES", "2"))
DEFAULT_MAX_CHARS = int(os.getenv("PDF_MAX_CHARS", "9000"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLAMA_MAX_TOKENS", "4800"))
PAGE_SNIPPET_HEAD_CHARS = int(os.getenv("PAGE_SNIPPET_HEAD_CHARS", "1400"))
PAGE_SNIPPET_TAIL_CHARS = int(os.getenv("PAGE_SNIPPET_TAIL_CHARS", "320"))
PATENT_WINDOW_BACK = int(os.getenv("PATENT_WINDOW_BACK", "1"))
PATENT_WINDOW_FORWARD = int(os.getenv("PATENT_WINDOW_FORWARD", "2"))
PATENT_EXPANSION_STEP = int(os.getenv("PATENT_EXPANSION_STEP", "2"))
PATENT_EXPANSION_ATTEMPTS = int(os.getenv("PATENT_EXPANSION_ATTEMPTS", "3"))
PATENT_MIN_RESULT_CHARS = int(os.getenv("PATENT_MIN_RESULT_CHARS", "160"))
PATENT_ALLOW_FULL_DOC_FALLBACK = os.getenv("PATENT_ALLOW_FULL_DOC_FALLBACK", "1") == "1"
PATENT_FULL_DOC_MAX_CHARS = int(os.getenv("PATENT_FULL_DOC_MAX_CHARS", "120000"))
PATENT_MAX_CANDIDATES = int(os.getenv("PATENT_MAX_CANDIDATES", "6"))
BUILD_ID = "claims-anchor-v6r-backend1"

TASK_PROFILES = {
    "abstract": {
        "label": "Paper Abstract",
        "pipeline": "front_text",
        "prompt": """this is text extracted from a scientific paper. find the paper's abstract. the abstract may or may not be explicitly labeled. output only the abstract text and nothing else.

important:
- choose the single best contiguous abstract-like block from the text.
- the abstract is usually one continuous paragraph or a small set of consecutive paragraphs near the beginning.
- do not start from a partial sentence.
- do not include title, author list, affiliations, copyright text, keywords, section headings, introduction text, body text, or text from a neighboring column.
- if some nearby text is clearly unrelated body text, ignore it and return only the coherent abstract block.
- do not rewrite or summarize; copy the abstract text from the input as faithfully as possible.

if no abstract is present, output exactly: NOT_FOUND""",
        "message_style": "single_user",
        "max_pages": DEFAULT_MAX_PAGES,
        "max_chars": DEFAULT_MAX_CHARS,
        "max_tokens": DEFAULT_MAX_TOKENS,
    },
    "first_claim": {
        "label": "Patent First Claim",
        "pipeline": "patent_claim",
        "prompt": """You are given text extracted from the beginning of a patent document.
Extract only claim 1 from the patent text below.
Claim 1 may begin on one page and continue onto later pages.
Capture everything from the start of claim 1 until immediately before claim 2.
The text may be badly formatted or split across multiple lines.
Reconstruct claim 1 as faithfully as possible from the provided text.
Output only claim 1 and nothing else.
If the first claim is not present in the provided text, return exactly: NOT_FOUND""",
        "max_tokens": int(os.getenv("PATENT_MAX_TOKENS", "9600")),
    },
}

app = Flask(__name__)
_RESOLVED_LLAMA_MODEL: Optional[str] = None
DEBUG_DUMP_DIR = Path(os.getenv("PDF_PARSER_DEBUG_DIR", "/tmp/pdf-parser-debug"))


def debug_log(message: str) -> None:
    if LLAMA_DEBUG:
        print(f"[pdf-parser] {message}", file=sys.stderr, flush=True)


def dump_debug_text(name: str, text: str) -> None:
    if not LLAMA_DEBUG:
        return
    DEBUG_DUMP_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DUMP_DIR / name
    path.write_text(text, encoding="utf-8")
    debug_log(f"wrote debug text to {path}")


def resolve_llama_model() -> str:
    global _RESOLVED_LLAMA_MODEL

    if LLAMA_MODEL:
        return LLAMA_MODEL
    if _RESOLVED_LLAMA_MODEL:
        return _RESOLVED_LLAMA_MODEL

    response = requests.get(f"{LLAMA_BASE_URL}/v1/models", timeout=10)
    response.raise_for_status()
    data = response.json()
    models = data.get("data") or []
    if not models:
        raise ValueError("llama-server did not report any models from /v1/models.")

    model_id = str(models[0].get("id", "")).strip()
    if not model_id:
        raise ValueError("llama-server returned a model entry without an id.")

    _RESOLVED_LLAMA_MODEL = model_id
    debug_log(f"Auto-detected llama model: {_RESOLVED_LLAMA_MODEL}")
    return _RESOLVED_LLAMA_MODEL


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_blocks(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    rect = page.rect
    page_width = float(rect.width)
    mid_x = float(rect.x0 + (rect.width / 2))
    text_blocks: list[tuple[int, float, float, str]] = []

    for block in blocks:
        if len(block) < 5:
            continue
        x0, y0, x1, _y1, text = block[:5]
        block_type = block[6] if len(block) > 6 else 0
        if block_type != 0:
            continue
        if not isinstance(text, str):
            continue
        cleaned = normalize_text(text)
        if not cleaned:
            continue
        x0f = float(x0)
        x1f = float(x1)
        block_width = max(0.0, x1f - x0f)

        if x0f <= page_width * 0.18 and block_width >= page_width * 0.42:
            column_order = 0
        elif x0f < mid_x:
            column_order = 1
        else:
            column_order = 2

        text_blocks.append((column_order, float(y0), x0f, cleaned))

    text_blocks.sort(key=lambda item: (item[0], round(item[1], 1), round(item[2], 1)))
    return normalize_text("\n\n".join(text for _col, _y0, _x0, text in text_blocks))


def extract_page_text(page: fitz.Page, two_column: bool = False, use_blocks: bool = False) -> str:
    if not two_column:
        if use_blocks:
            return extract_text_blocks(page)
        return normalize_text(page.get_text("text", sort=True))

    rect = page.rect
    mid_x = rect.x0 + (rect.width / 2)
    left_clip = fitz.Rect(rect.x0, rect.y0, mid_x, rect.y1)
    right_clip = fitz.Rect(mid_x, rect.y0, rect.x1, rect.y1)
    left_text = normalize_text(page.get_text("text", sort=True, clip=left_clip))
    right_text = normalize_text(page.get_text("text", sort=True, clip=right_clip))
    return normalize_text("\n".join(part for part in (left_text, right_text) if part))


def extract_pdf_pages(
    pdf_bytes: bytes,
    two_column: bool = False,
    use_blocks: bool = False,
) -> list[str]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
        pages = []
        for page in document:
            page_text = extract_page_text(page, two_column=two_column, use_blocks=use_blocks)
            pages.append(page_text)

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    return pages


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0]
    return text


def trim_abstract_region(text: str) -> str:
    heading_patterns = [
        r"(?im)^\s*(?:\d+[\.\)]\s*)?introduction\s*$",
        r"(?im)^\s*(?:\d+[\.\)]\s*)?background\s*$",
        r"(?im)^\s*(?:\d+[\.\)]\s*)?(?:materials?\s+and\s+methods?|methods?)\s*$",
        r"(?im)^\s*(?:\d+[\.\)]\s*)?results?\s*$",
        r"(?im)^\s*(?:\d+[\.\)]\s*)?discussion\s*$",
        r"(?im)^\s*(?:\d+[\.\)]\s*)?conclusions?\s*$",
    ]

    cut_index: Optional[int] = None
    for pattern in heading_patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        if cut_index is None or match.start() < cut_index:
            cut_index = match.start()

    if cut_index is not None and cut_index > 0:
        return text[:cut_index].strip()
    return text


def extract_front_text(page_texts: list[str], max_pages: int, max_chars: int) -> str:
    visible_pages = [text for text in page_texts[:max_pages] if text]
    if not visible_pages:
        raise ValueError("No text could be extracted from the selected PDF pages.")
    front_text = normalize_text("\n\n".join(visible_pages))
    front_text = trim_abstract_region(front_text)
    return truncate_text(front_text, max_chars)


def get_task_config(task_name: str, custom_prompt: str) -> dict[str, Any]:
    task = TASK_PROFILES.get(task_name, TASK_PROFILES["abstract"]).copy()
    if custom_prompt.strip():
        task["prompt"] = custom_prompt.strip()
    return task


def collect_text_fragments(value: Any, skip_reasoning: bool) -> list[str]:
    items: list[Any]
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    text_parts: list[str] = []
    for item in items:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            continue

        item_type = str(item.get("type", "")).lower()
        if skip_reasoning and item_type in {"reasoning", "reasoning_content"}:
            continue
        if not skip_reasoning and item_type not in {"reasoning", "reasoning_content"}:
            continue

        for key in ("text", "content", "value"):
            inner_value = item.get(key)
            if isinstance(inner_value, str):
                text_parts.append(inner_value)
                break
    return text_parts


def extract_choice_text(choice: dict[str, Any]) -> tuple[str, str]:
    payload = choice.get("delta") or choice.get("message") or {}
    content = payload.get("content", "")
    reasoning_content = payload.get("reasoning_content", "")

    visible_content = "".join(collect_text_fragments(content, skip_reasoning=True))
    embedded_reasoning = "".join(collect_text_fragments(content, skip_reasoning=False))
    reasoning_text = ""

    if isinstance(reasoning_content, (list, dict, str)):
        reasoning_text = "".join(collect_text_fragments(reasoning_content, skip_reasoning=False))
    if not reasoning_text:
        reasoning_text = embedded_reasoning
    if not visible_content and isinstance(choice.get("text"), str):
        visible_content = choice["text"]

    return visible_content, reasoning_text


def extract_with_llama(
    paper_text: str,
    system_prompt: str,
    max_tokens: int,
    enable_thinking: Optional[bool] = None,
    user_only: bool = False,
) -> tuple[str, str]:
    if enable_thinking is None:
        enable_thinking = LLAMA_ENABLE_THINKING

    messages: list[dict[str, str]]
    if user_only:
        messages = [{"role": "user", "content": paper_text}]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": paper_text},
        ]

    model_name = resolve_llama_model()
    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "messages": messages,
    }
    payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    response = requests.post(
        f"{LLAMA_BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Accept": "text/event-stream"},
        timeout=LLAMA_TIMEOUT,
        stream=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    debug_log(
        f"chat request model={model_name} stream=True thinking={enable_thinking} user_only={user_only} "
        f"content_type={content_type!r} prompt_chars={len(system_prompt)} text_chars={len(paper_text)} "
        f"max_tokens={max_tokens}"
    )
    finish_reason = "unknown"
    content_types: list[str] = []
    message_keys: list[str] = []
    has_reasoning = False

    if content_type.startswith("text/event-stream"):
        visible_parts: list[str] = []
        reasoning_parts: list[str] = []

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            if not raw_line.startswith("data:"):
                continue
            data_line = raw_line[5:].strip()
            if data_line == "[DONE]":
                break
            try:
                event = json.loads(data_line)
            except json.JSONDecodeError:
                continue

            choices = event.get("choices") or []
            if not choices:
                continue

            choice = choices[0]
            delta = choice.get("delta") or choice.get("message") or {}
            if isinstance(delta, dict):
                message_keys = sorted(set(message_keys) | set(delta.keys()))
                if isinstance(delta.get("content"), list):
                    for item in delta["content"]:
                        if isinstance(item, dict) and item.get("type") is not None:
                            content_types.append(str(item.get("type")))
                has_reasoning = has_reasoning or bool(delta.get("reasoning_content"))

            visible_text, reasoning_text = extract_choice_text(choice)
            if visible_text:
                visible_parts.append(visible_text)
            if reasoning_text:
                reasoning_parts.append(reasoning_text)

            if choice.get("finish_reason") is not None:
                finish_reason = choice.get("finish_reason", finish_reason)

        result = "".join(visible_parts).strip().strip("`").strip()
        reasoning_text = "".join(reasoning_parts).strip()
        debug_log(
            f"sse parse complete visible_chars={len(result)} reasoning_chars={len(reasoning_text)} "
            f"finish_reason={finish_reason} message_keys={message_keys} content_types={content_types}"
        )
    else:
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("Model response did not include any choices.")

        choice = choices[0]
        message = choice.get("message") or {}
        if isinstance(message, dict):
            message_keys = sorted(message.keys())
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") is not None:
                        content_types.append(str(item.get("type")))
            has_reasoning = bool(message.get("reasoning_content")) or "reasoning_content" in content_types

        visible_text, reasoning_text = extract_choice_text(choice)
        result = visible_text.strip().strip("`").strip()
        reasoning_text = reasoning_text.strip()
        finish_reason = choice.get("finish_reason", "unknown")
        debug_log(
            f"json parse complete visible_chars={len(result)} reasoning_chars={len(reasoning_text)} "
            f"finish_reason={finish_reason} message_keys={message_keys} content_types={content_types}"
        )

    if not result:
        raise ValueError(
            "Model returned an empty response. "
            f"model={model_name}; "
            f"finish_reason={finish_reason}; "
            f"message_keys={message_keys}; "
            f"content_types={content_types}; "
            f"has_reasoning_content={has_reasoning}; "
            f"LLAMA_ENABLE_THINKING={enable_thinking}"
        )

    return result, reasoning_text


def parse_uploaded_pdf(
    uploaded_file: Any, selected_task: str, custom_prompt: str
) -> dict[str, Any]:
    if uploaded_file is None or uploaded_file.filename == "":
        raise ValueError("Choose a PDF file first.")

    task = get_task_config(selected_task, custom_prompt)
    pdf_bytes = uploaded_file.read()
    reasoning_trace = ""
    if task["pipeline"] == "patent_claim":
        page_texts = extract_pdf_pages(pdf_bytes, two_column=True)
        result, result_meta = extract_patent_claim(page_texts, task)
        chars_examined = sum(len(page_texts[page - 1]) for page in result_meta["used_pages"])
    else:
        page_texts = extract_pdf_pages(pdf_bytes, use_blocks=PAPER_USE_BLOCKS)
        paper_text = extract_front_text(page_texts, task["max_pages"], task["max_chars"])
        dump_debug_text("paper_front_text.txt", paper_text)
        if task.get("message_style") == "single_user":
            user_message = f"{task['prompt']}\n\n{paper_text}"
            dump_debug_text("paper_user_message.txt", user_message)
            result, reasoning_trace = extract_with_llama(
                user_message,
                "",
                task["max_tokens"],
                user_only=True,
            )
        else:
            dump_debug_text("paper_system_prompt.txt", task["prompt"])
            dump_debug_text("paper_user_message.txt", paper_text)
            result, reasoning_trace = extract_with_llama(paper_text, task["prompt"], task["max_tokens"])
        result_meta = {
            "candidate_pages": list(range(1, min(task["max_pages"], len(page_texts)) + 1)),
            "used_pages": list(range(1, min(task["max_pages"], len(page_texts)) + 1)),
            "mode": "front_text",
            "attempts": [],
        }
        chars_examined = len(paper_text)

    return {
        "task_id": selected_task,
        "task_label": task["label"],
        "filename": uploaded_file.filename,
        "result": result,
        "display_result": format_display_result(selected_task, result),
        "reasoning_trace": reasoning_trace,
        "result_meta": result_meta,
        "chars_examined": chars_examined,
    }


def summarize_page(page_text: str) -> str:
    if not page_text:
        return "[no extractable text]"
    if len(page_text) <= PAGE_SNIPPET_HEAD_CHARS + PAGE_SNIPPET_TAIL_CHARS:
        return page_text

    head = page_text[:PAGE_SNIPPET_HEAD_CHARS].strip()
    tail = page_text[-PAGE_SNIPPET_TAIL_CHARS :].strip()
    return f"{head}\n...\n{tail}"


def build_page_summary_prompt(page_texts: list[str]) -> str:
    blocks = []
    for page_number, page_text in enumerate(page_texts, start=1):
        blocks.append(f"[Page {page_number}]\n{summarize_page(page_text)}")
    return "\n\n".join(blocks)


def heuristic_claim_candidates(page_texts: list[str]) -> list[int]:
    total_pages = len(page_texts)
    later_half_start = max(1, total_pages // 2)
    scored_pages: list[tuple[int, int]] = []
    for page_number, page_text in enumerate(page_texts, start=1):
        lower_text = page_text.lower()
        score = 0
        if "we claim:" in lower_text or "we claim" in lower_text:
            score += 18
        if "what is claimed is" in lower_text:
            score += 12
        if "the invention claimed is" in lower_text:
            score += 12
        if re.search(r"(^|\n)\s*claims?\s*$", lower_text, flags=re.MULTILINE):
            score += 8
        if re.search(r"(^|\n)\s*claim\s+1[\.:)]", lower_text, flags=re.MULTILINE):
            score += 7
        if re.search(r"(^|\n)\s*1[\.:)]\s", lower_text, flags=re.MULTILINE):
            score += 5
        if re.search(r"(^|\n)\s*2[\.:)]\s", lower_text, flags=re.MULTILINE):
            score += 2
        if "claims priority" in lower_text or "cross reference to related" in lower_text:
            score -= 10
        if page_number >= later_half_start:
            score += 6
        if page_number >= 4:
            score += 1
        if score > 0:
            scored_pages.append((score, page_number))

    if not scored_pages:
        return [1]

    scored_pages.sort(key=lambda item: (-item[0], item[1]))
    return [page_number for _, page_number in scored_pages[:5]]


def parse_page_numbers(raw_text: str, page_count: int) -> list[int]:
    page_numbers = []
    for match in re.findall(r"\d+", raw_text):
        value = int(match)
        if 1 <= value <= page_count and value not in page_numbers:
            page_numbers.append(value)
    return page_numbers


def rank_claim_pages_with_llama(page_texts: list[str]) -> list[int]:
    ranking_prompt = """You are given page-by-page extracted text snippets from a patent.
Identify the pages most likely to contain the beginning of claim 1.
Prefer pages near the claims section, not early pages that only mention patent claims or priority claims.
Return only a comma-separated list of up to 5 page numbers, ordered from best to worst.
Do not add any explanation or extra words."""
    ranking_input = build_page_summary_prompt(page_texts)
    raw_ranking, _ = extract_with_llama(
        ranking_input,
        ranking_prompt,
        max_tokens=80,
        enable_thinking=False,
    )
    ranked_pages = parse_page_numbers(raw_ranking, len(page_texts))
    heuristic_pages = heuristic_claim_candidates(page_texts)

    merged_pages = []
    for page_number in heuristic_pages + ranked_pages:
        if page_number not in merged_pages:
            merged_pages.append(page_number)
    return merged_pages or [1]


def format_page_window(page_texts: list[str], page_numbers: list[int]) -> str:
    blocks = []
    for page_number in page_numbers:
        blocks.append(f"[[PAGE {page_number}]]\n{page_texts[page_number - 1]}")
    return "\n\n".join(blocks)


def format_raw_window(page_texts: list[str], page_numbers: list[int]) -> str:
    return "\n\n".join(page_texts[page_number - 1] for page_number in page_numbers)


def clean_patent_claim_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"(?m)^\s*US\s+\d[\d,\s]*$", "", text)
    text = re.sub(r"(?m)^\s*,?\d{2,4}\s+B\d+\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d{1,3}\s+\d{1,3}\s*$", "", text)
    text = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", text)
    text = re.sub(r"(?m)\s+\d{1,3}\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped in {"B2", "B1", "A1", "A2"}:
            continue
        if re.fullmatch(r"\d{1,3}", stripped):
            continue
        if re.fullmatch(r"PF\s+[A-Z0-9 ]+", stripped):
            continue
        lines.append(line.rstrip())

    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"(?m)^\s*US\s+\d[\d,\s]*", "", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def is_formulaish_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if lowered.startswith("formula "):
        return True
    if len(stripped) <= 18 and len(stripped.split()) <= 4:
        return True
    if re.fullmatch(r"[A-Za-z0-9(),./=\-—+\s]+", stripped) and len(stripped) <= 20:
        return True
    return False


def prettify_patent_display(text: str) -> str:
    output: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            output.append(" ".join(paragraph))
            paragraph.clear()

    for raw_line in text.splitlines():
        line = re.sub(r"\s{2,}", " ", raw_line.strip())
        if not line:
            flush_paragraph()
            if output and output[-1] != "":
                output.append("")
            continue
        if is_formulaish_line(line):
            flush_paragraph()
            output.append(line)
            continue
        paragraph.append(line)

    flush_paragraph()

    pretty_lines = []
    previous_blank = False
    for line in output:
        if line == "":
            if not previous_blank:
                pretty_lines.append(line)
            previous_blank = True
            continue
        pretty_lines.append(line)
        previous_blank = False

    return "\n".join(pretty_lines).strip()


def format_display_result(task_id: str, result: str) -> str:
    if task_id == "first_claim":
        return prettify_patent_display(result)
    return result


def has_claim_anchor(text: str) -> bool:
    lower_text = text.lower()
    return any(
        marker in lower_text
        for marker in ("we claim", "what is claimed is", "the invention claimed is")
    )


def has_claim_one_start(text: str) -> bool:
    return re.search(r"(^|\n)\s*1[\.)]\s+", text, flags=re.MULTILINE) is not None


def has_claim_two_start(text: str) -> bool:
    return re.search(r"(^|\n)\s*2[\.)]\s+", text, flags=re.MULTILINE) is not None


def find_claim_anchor_pages(page_texts: list[str]) -> list[int]:
    anchors = []
    for page_number, page_text in enumerate(page_texts, start=1):
        if has_claim_anchor(page_text):
            anchors.append(page_number)
    return anchors


def extract_claim_one_from_text(text: str) -> Optional[str]:
    anchor_match = re.search(
        r"(we claim|what is claimed is|the invention claimed is)\s*:?",
        text,
        flags=re.IGNORECASE,
    )
    search_text = text[anchor_match.end() :] if anchor_match else text

    start_match = re.search(r"(^|\n)\s*1[\.)]\s+", search_text, flags=re.MULTILINE)
    if not start_match:
        return None

    extracted = search_text[start_match.start() :].strip()
    end_match = re.search(r"(^|\n)\s*2[\.)]\s+", extracted, flags=re.MULTILINE)
    if end_match:
        extracted = extracted[: end_match.start()].strip()
    return clean_patent_claim_text(extracted)


def strip_after_claim_two(text: str) -> str:
    match = re.search(r"(^|\n)\s*2[\.)]\s", text, flags=re.MULTILINE)
    if not match:
        return clean_patent_claim_text(text)
    return clean_patent_claim_text(text[: match.start()].strip())


def source_window_supports_claim_extraction(source_text: str) -> bool:
    return has_claim_anchor(source_text) or has_claim_two_start(source_text)


def is_claim_result_strong(result: str, source_text: str) -> bool:
    cleaned = strip_after_claim_two(result)
    if cleaned == "NOT_FOUND":
        return False
    if not source_window_supports_claim_extraction(source_text):
        return False
    if len(cleaned) < PATENT_MIN_RESULT_CHARS:
        return False
    if not has_claim_one_start(cleaned) and not re.search(r"^\s*a\s", cleaned.lower()):
        return False
    if cleaned.endswith("."):
        return True
    if len(cleaned) >= PATENT_MIN_RESULT_CHARS * 2:
        return True
    return False


def try_claim_window(
    page_texts: list[str], task: dict[str, Any], best_page: int
) -> tuple[Optional[str], list[list[int]]]:
    start_page = max(1, best_page - PATENT_WINDOW_BACK)
    end_page = min(len(page_texts), best_page + PATENT_WINDOW_FORWARD)
    attempts: list[list[int]] = []

    for _ in range(PATENT_EXPANSION_ATTEMPTS):
        window_pages = list(range(start_page, end_page + 1))
        raw_window_text = format_raw_window(page_texts, window_pages)
        window_text = format_page_window(page_texts, window_pages)

        direct_claim = extract_claim_one_from_text(raw_window_text)
        if direct_claim and is_claim_result_strong(direct_claim, raw_window_text):
            attempts.append(window_pages)
            return direct_claim, attempts

        result = strip_after_claim_two(
            extract_with_llama(
                window_text,
                task["prompt"],
                task["max_tokens"],
                enable_thinking=False,
            )[0]
        )
        attempts.append(window_pages)
        if is_claim_result_strong(result, raw_window_text):
            return result, attempts

        if end_page < len(page_texts):
            end_page = min(len(page_texts), end_page + PATENT_EXPANSION_STEP)
            continue
        if start_page > 1:
            start_page = max(1, start_page - PATENT_EXPANSION_STEP)
            continue
        break

    return None, attempts


def extract_patent_claim(page_texts: list[str], task: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    anchor_pages = find_claim_anchor_pages(page_texts)
    ranked_pages = rank_claim_pages_with_llama(page_texts)
    ordered_candidates = []
    for page_number in anchor_pages + ranked_pages:
        if page_number not in ordered_candidates:
            ordered_candidates.append(page_number)
    attempts: list[list[int]] = []

    for best_page in ordered_candidates[:PATENT_MAX_CANDIDATES]:
        result, candidate_attempts = try_claim_window(page_texts, task, best_page)
        attempts.extend(candidate_attempts)
        if result is None:
            continue

        return result, {
            "candidate_pages": ordered_candidates[:PATENT_MAX_CANDIDATES],
            "used_pages": candidate_attempts[-1],
            "mode": "focused_pages",
            "attempts": attempts,
        }

    if PATENT_ALLOW_FULL_DOC_FALLBACK:
        full_text = normalize_text(format_page_window(page_texts, list(range(1, len(page_texts) + 1))))
        if len(full_text) <= PATENT_FULL_DOC_MAX_CHARS:
            result = strip_after_claim_two(
                extract_with_llama(
                    full_text,
                    task["prompt"],
                    task["max_tokens"],
                    enable_thinking=False,
                )[0]
            )
            return result, {
                "candidate_pages": ordered_candidates[:PATENT_MAX_CANDIDATES],
                "used_pages": list(range(1, len(page_texts) + 1)),
                "mode": "full_document_fallback",
                "attempts": attempts,
            }

    last_pages = attempts[-1] if attempts else [ordered_candidates[0]]
    return "NOT_FOUND", {
        "candidate_pages": ordered_candidates[:PATENT_MAX_CANDIDATES],
        "used_pages": last_pages,
        "mode": "focused_pages",
        "attempts": attempts,
    }


@app.get("/")
def index() -> str:
    return render_template(
        "index.html",
        tasks=TASK_PROFILES,
        selected_task="abstract",
        custom_prompt="",
        build_id=BUILD_ID,
    )


@app.get("/parse")
def parse_redirect() -> Any:
    return redirect(url_for("index"))


@app.post("/api/parse")
def parse_pdf_api() -> Any:
    uploaded_file = request.files.get("pdf")
    selected_task = request.form.get("task", "abstract")
    custom_prompt = request.form.get("custom_prompt", "")

    try:
        payload = parse_uploaded_pdf(uploaded_file, selected_task, custom_prompt)
    except requests.RequestException as exc:
        return (
            jsonify(
                {
                    "error": "Failed to reach llama-server.",
                    "details": str(exc),
                    "build_id": BUILD_ID,
                }
            ),
            502,
        )
    except Exception as exc:
        return jsonify({"error": "Could not parse this PDF.", "details": str(exc), "build_id": BUILD_ID}), 400

    payload["build_id"] = BUILD_ID
    return jsonify(payload)


@app.post("/parse")
def parse_pdf() -> str:
    uploaded_file = request.files.get("pdf")
    selected_task = request.form.get("task", "abstract")
    custom_prompt = request.form.get("custom_prompt", "")
    if uploaded_file is None or uploaded_file.filename == "":
        return (
            render_template(
                "index.html",
                error="Choose a PDF file first.",
                tasks=TASK_PROFILES,
                selected_task=selected_task,
                custom_prompt=custom_prompt,
                build_id=BUILD_ID,
            ),
            400,
        )

    try:
        payload = parse_uploaded_pdf(uploaded_file, selected_task, custom_prompt)
    except requests.RequestException as exc:
        message = (
            "Failed to reach llama-server. Make sure it is running and the "
            "LLAMA_BASE_URL / LLAMA_MODEL settings are correct."
        )
        return (
            render_template(
                "index.html",
                error=message,
                details=str(exc),
                tasks=TASK_PROFILES,
                selected_task=selected_task,
                custom_prompt=custom_prompt,
                build_id=BUILD_ID,
            ),
            502,
        )
    except Exception as exc:
        return (
            render_template(
                "index.html",
                error="Could not parse this PDF.",
                details=str(exc),
                tasks=TASK_PROFILES,
                selected_task=selected_task,
                custom_prompt=custom_prompt,
                build_id=BUILD_ID,
            ),
            400,
        )

    return render_template(
        "index.html",
        tasks=TASK_PROFILES,
        selected_task=selected_task,
        custom_prompt=custom_prompt,
        result=payload["result"],
        display_result=payload["display_result"],
        result_label=payload["task_label"],
        filename=payload["filename"],
        chars_examined=payload["chars_examined"],
        result_meta=payload["result_meta"],
        build_id=BUILD_ID,
    )


if __name__ == "__main__":
    app.run(host=BACKEND_HOST, port=BACKEND_PORT, debug=False)
