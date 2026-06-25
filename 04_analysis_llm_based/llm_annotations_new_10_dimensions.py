#!/usr/bin/env python3
"""
Annotate English museum-robot dialogues on the 10 Part A/B dimensions using
all prompt files in 04_analysis_llm_based/prompts/ and all models listed in
configs/llm_apis_ready.json.

Key features:
- Supports the robot images saved in data/assets/robot_images/.
- Uses image input for likely multimodal models when --part-b-mode auto.
- Falls back to standardized text descriptions for text-only models.
- Fills common {{placeholders}} in prompt files if present.
- Saves raw responses and parsed A1-B4 score columns.
- Uses a JSONL cache so interrupted runs can be resumed.

Run from repo root:
    python 04_analysis_llm_based/llm_annotations_new_10_dimensions.py

Smoke test:
    python 04_analysis_llm_based/llm_annotations_new_10_dimensions.py --max-dialogs 2

Force actual image input, only recommended for vision-capable models:
    python 04_analysis_llm_based/llm_annotations_new_10_dimensions.py --part-b-mode image

Use text-only Part B descriptions for all models:
    python 04_analysis_llm_based/llm_annotations_new_10_dimensions.py --part-b-mode description
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional but recommended for cleaner progress reporting
    tqdm = None


# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

with (REPO_ROOT / "configs" / "paths.json").open("r", encoding="utf-8") as _f:
    PATHS = {k: REPO_ROOT / v for k, v in json.load(_f).items() if not k.startswith("_")}

load_dotenv(dotenv_path=PATHS["env_file"], override=True)

INPUT_CSV = PATHS["dialogs_for_annotation_en"]
CACHE_PATH = PATHS["llm_new_annotations_cache"]
OUTPUT_CSV = PATHS["llm_new_annotations_long"]

ROBOT_IMAGE_DIR = REPO_ROOT / "data" / "assets" / "robot_images"
PART_B_IMAGES = {
    "A": ROBOT_IMAGE_DIR / "block_b_image_condition_a_willi.png",
    "B": ROBOT_IMAGE_DIR / "block_b_image_condition_b_wv34.png",
}

# These descriptions are only used when the selected model is text-only or when
# --part-b-mode description is used. For the closest match to the human study,
# use --part-b-mode image with vision-capable models.
PART_B_DESCRIPTIONS = {
    "A": (
        "Part B context: the robot image for Condition A (Willi) is used. "
        "This is the condition with facial expressions/gestures. "
        "Use this context only for B1-B4, not for A1-A6."
    ),
    "B": (
        "Part B context: the robot image for Condition B (WV-34) is used. "
        "This is the condition with a flat metallic voice and no expressions. "
        "Use this context only for B1-B4, not for A1-A6."
    ),
}

DIMENSION_KEYS = [
    "A1_user_engagement_enjoyment",
    "A2_conversation_flow_coherence",
    "A3_interaction_clarity_habitability",
    "A4_repair_recovery_quality",
    "A5_response_appropriateness",
    "A6_social_interaction_quality",
    "B1_anthropomorphism_human_likeness",
    "B2_animacy_lifelikeness",
    "B3_likeability_pleasantness",
    "B4_perceived_intelligence_competence",
]

PROVIDER_ENV = {
    "nhr": {"api_key": "NHR_API_KEY", "base_url": "NHR_BASE_URL"},
    "gwdg": {"api_key": "GWDG_API_KEY", "base_url": "GWDG_BASE_URL"},
    "openai": {"api_key": "OPENAI_API_KEY", "base_url": None},
}

CACHE_VERSION = "v2_part_b_context_and_parsed_scores"


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------


def load_models(ready_models_path: Path) -> list[tuple[str, str]]:
    """Return [(provider, model_id), ...] for every model in llm_apis_ready.json."""
    with ready_models_path.open("r", encoding="utf-8") as f:
        ready = json.load(f)

    models: list[tuple[str, str]] = []
    for provider, model_map in ready.items():
        if provider.startswith("_"):
            continue
        if provider not in PROVIDER_ENV:
            print(f"Warning: provider {provider!r} is not configured in PROVIDER_ENV; skipping.")
            continue
        for model_id in model_map:
            models.append((provider, model_id))
    return models



def load_prompts(prompts_dir: Path) -> list[tuple[str, str]]:
    """Return [(prompt_name, prompt_text), ...] for every *.txt file in prompts/."""
    prompts: list[tuple[str, str]] = []
    for path in sorted(prompts_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            prompts.append((path.stem, text))
    return prompts



def load_dialogs(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))



def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = row.get("key")
                if key:
                    cache[key] = row
            except json.JSONDecodeError as exc:
                print(f"Warning: could not parse cache line {line_no}: {exc}")
    return cache



def append_cache(cache_path: Path, row: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# -----------------------------------------------------------------------------
# Part B image/context helpers
# -----------------------------------------------------------------------------


def normalise_condition(raw: Any) -> str | None:
    """Map condition labels from the CSV to 'A' or 'B'."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None

    if text in {"a", "condition a", "condition_a"}:
        return "A"
    if text in {"b", "condition b", "condition_b"}:
        return "B"

    if "willi" in text or "condition a" in text or "condition_a" in text:
        return "A"
    if "wv" in text or "wv-34" in text or "condition b" in text or "condition_b" in text:
        return "B"

    return None



def likely_supports_images(provider: str, model: str) -> bool:
    """
    Heuristic for --part-b-mode auto.

    Keep this conservative. If a model is not detected here but you know it is
    multimodal, either add a keyword below or run with --part-b-mode image.
    """
    model_l = model.lower()
    vision_keywords = [
        "vision",
        "vl",
        "qwen-vl",
        "qwen2-vl",
        "qwen2.5-vl",
        "llava",
        "gemini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "o3",
        "o4",
    ]
    return any(keyword in model_l for keyword in vision_keywords)



def image_to_data_uri(image_path: Path) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"Part B image not found: {image_path}")

    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime = "image/png"
    elif suffix in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    elif suffix == ".webp":
        mime = "image/webp"
    else:
        raise ValueError(f"Unsupported image type for {image_path}; use PNG, JPG, JPEG, or WEBP.")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"



def choose_part_b_input(
    dialog: dict[str, Any],
    provider: str,
    model: str,
    part_b_mode: str,
) -> tuple[str, Path | None, str | None]:
    """
    Return (part_b_input_type, image_path, part_b_text).

    part_b_input_type is one of: image, description, none, unknown_condition.
    """
    condition = normalise_condition(
        dialog.get("condition_hidden")
        or dialog.get("condition")
        or dialog.get("condition_source")
    )

    if condition not in {"A", "B"}:
        return (
            "unknown_condition",
            None,
            "Part B context is unavailable because the condition could not be determined. Return null for B1-B4.",
        )

    if part_b_mode == "none":
        return "none", None, "Part B context was intentionally not provided. Return null for B1-B4."

    use_image = part_b_mode == "image" or (
        part_b_mode == "auto" and likely_supports_images(provider, model)
    )

    if use_image:
        image_path = PART_B_IMAGES[condition]
        return (
            "image",
            image_path,
            "Part B context: the corresponding robot image is attached. Use it only for B1-B4, not for A1-A6.",
        )

    return "description", None, PART_B_DESCRIPTIONS[condition]


# -----------------------------------------------------------------------------
# Prompt/user-message construction
# -----------------------------------------------------------------------------


def render_template(text: str, context: dict[str, Any]) -> str:
    """Replace simple {{key}} placeholders in a prompt file."""
    rendered = text
    for key, value in context.items():
        rendered = rendered.replace("{{" + key + "}}", "" if value is None else str(value))
    return rendered



def contains_input_placeholders(prompt_text: str) -> bool:
    placeholders = [
        "{{dialog_id}}",
        "{{language}}",
        "{{topic_main}}",
        "{{dialogue_for_annotation}}",
        "{{part_b_context}}",
    ]
    return any(placeholder in prompt_text for placeholder in placeholders)



def build_user_text(context: dict[str, Any], prompt_contains_placeholders: bool) -> str:
    """
    Build the user message.

    If the prompt file already contains and receives the full rendered input via
    placeholders, keep the user message short to avoid sending the transcript twice.
    Otherwise, provide a standard input block here.
    """
    if prompt_contains_placeholders:
        return (
            "Please annotate the provided dialogue and Part B context. "
            "Return valid JSON only, following the requested schema."
        )

    return f"""Input to annotate

dialog_id: {context.get('dialog_id', '')}
language: {context.get('language', '')}
topic_main: {context.get('topic_main', '')}

Part A dialogue transcript:
{context.get('dialogue_for_annotation', '')}

Part B context:
{context.get('part_b_context', '')}

Return valid JSON only."""



def make_messages(system_prompt: str, user_text: str, image_path: Path | None) -> list[dict[str, Any]]:
    if image_path is None:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}},
            ],
        },
    ]


# -----------------------------------------------------------------------------
# Model calling
# -----------------------------------------------------------------------------

_clients: dict[str, Any] = {}



def get_client(provider: str):
    if provider in _clients:
        return _clients[provider]

    from openai import OpenAI

    env = PROVIDER_ENV[provider]
    api_key = os.getenv(env["api_key"])
    base_url = os.getenv(env["base_url"]) if env["base_url"] else None
    if not api_key:
        raise RuntimeError(f"{env['api_key']} is not set. Add it to .env before running.")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120.0)
    _clients[provider] = client
    return client



def call_model(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_retries: int,
    retry_base_seconds: float,
    temperature: float,
) -> dict[str, Any]:
    client = get_client(provider)

    last_error: Exception | None = None
    t0 = time.time()
    for attempt in range(1, max_retries + 1):
        t0 = time.time()
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if provider == "openai":
                kwargs["max_completion_tokens"] = 2000
            else:
                kwargs["max_tokens"] = 2000

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content
            return {
                "status": "OK",
                "response_text": content.strip() if content else "",
                "error": None,
                "latency_s": round(time.time() - t0, 2),
            }
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_retries:
                break
            sleep_seconds = retry_base_seconds * (2 ** (attempt - 1))
            message = (
                f"[{provider}/{model}] attempt {attempt} failed "
                f"({type(exc).__name__}: {exc}). Retrying in {sleep_seconds:.1f}s..."
            )
            (tqdm.write if tqdm is not None else print)(message)
            time.sleep(sleep_seconds)

    return {
        "status": "FAIL",
        "response_text": None,
        "error": str(last_error),
        "latency_s": round(time.time() - t0, 2),
    }


# -----------------------------------------------------------------------------
# JSON parsing and score extraction
# -----------------------------------------------------------------------------


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()



def extract_json_object(text: str | None) -> tuple[dict[str, Any] | None, str, str | None]:
    """Return (parsed_json, parse_status, parse_error)."""
    if not text:
        return None, "empty", "empty response"

    cleaned = strip_code_fence(text)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed, "ok", None
        return None, "not_object", "parsed JSON is not an object"
    except json.JSONDecodeError:
        pass

    # Fallback: extract the first JSON-looking object from surrounding text.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "no_json_object", "no JSON object found"

    candidate = cleaned[start : end + 1]
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed, "ok_extracted", None
        return None, "not_object", "extracted JSON is not an object"
    except json.JSONDecodeError as exc:
        return None, "json_error", str(exc)



def score_to_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("score")
    if isinstance(value, str):
        value = value.strip()
        if value.lower() in {"null", "none", "na", "n/a", ""}:
            return None
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    if 1 <= score <= 5:
        return score
    return None



def confidence_or_none(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    conf = value.get("confidence")
    if conf is None:
        return None
    conf_str = str(conf).strip().lower()
    return conf_str if conf_str in {"low", "medium", "high"} else conf_str or None



def parse_annotation_response(response_text: str | None) -> dict[str, Any]:
    parsed, parse_status, parse_error = extract_json_object(response_text)

    out: dict[str, Any] = {
        "json_parse_status": parse_status,
        "json_parse_error": parse_error,
    }
    for key in DIMENSION_KEYS:
        out[key] = None
        out[f"{key}_confidence"] = None

    if parsed is None:
        return out

    # Support all prompt variants: {"scores": {...}}, {"ratings": {...}}, or top-level keys.
    container = parsed.get("scores") or parsed.get("ratings") or parsed
    if not isinstance(container, dict):
        out["json_parse_status"] = "json_wrong_schema"
        out["json_parse_error"] = "scores/ratings container is not an object"
        return out

    for key in DIMENSION_KEYS:
        raw_value = container.get(key)
        out[key] = score_to_int_or_none(raw_value)
        out[f"{key}_confidence"] = confidence_or_none(raw_value)

    return out


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def write_results_csv(cache: dict[str, dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "dialog_id",
        "language",
        "condition_hidden",
        "topic_main",
        "prompt_name",
        "provider",
        "model",
        "part_b_input_type",
        "part_b_image_path",
        "status",
        "response_text",
        "json_parse_status",
        "json_parse_error",
        *DIMENSION_KEYS,
        *[f"{key}_confidence" for key in DIMENSION_KEYS],
        "error",
        "latency_s",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        cache.values(),
        key=lambda r: (
            str(r.get("dialog_id", "")),
            str(r.get("prompt_name", "")),
            str(r.get("provider", "")),
            str(r.get("model", "")),
            str(r.get("part_b_input_type", "")),
        ),
    )

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-dialogs", type=int, default=None, help="Limit number of dialogues for smoke tests.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=5.0)
    parser.add_argument("--request-pause-seconds", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--overwrite-cache", action="store_true", help="Ignore existing cache and re-run everything.")
    parser.add_argument(
        "--part-b-mode",
        choices=["auto", "image", "description", "none"],
        default="auto",
        help=(
            "auto = attach image only for likely vision-capable models; "
            "image = always attach image; description = use text descriptions; none = B scores should be null."
        ),
    )
    args = parser.parse_args()

    dialogs = load_dialogs(INPUT_CSV)
    if args.max_dialogs is not None:
        dialogs = dialogs[: args.max_dialogs]

    prompts = load_prompts(PROMPTS_DIR)
    if not prompts:
        raise RuntimeError(f"No *.txt prompt files found in {PROMPTS_DIR}")

    models = load_models(PATHS["llm_apis_ready"])
    if not models:
        raise RuntimeError(f"No models found in {PATHS['llm_apis_ready']}")

    if args.part_b_mode in {"auto", "image"}:
        for condition, image_path in PART_B_IMAGES.items():
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Missing Part B image for condition {condition}: {image_path}\n"
                    "Expected location: data/assets/robot_images/"
                )

    cache = {} if args.overwrite_cache else load_cache(CACHE_PATH)

    print(f"Dialogues: {len(dialogs)} | Prompts: {len(prompts)} | Models: {len(models)}")
    print(f"Part B mode: {args.part_b_mode}")
    print(f"Total calls: {len(dialogs) * len(prompts) * len(models)} | Cached rows: {len(cache)}")

    jobs = [
        (dialog, prompt_name, prompt_text, provider, model)
        for dialog in dialogs
        for prompt_name, prompt_text in prompts
        for provider, model in models
    ]

    iterator = jobs
    progress = None
    if tqdm is not None:
        progress = tqdm(jobs, total=len(jobs), desc="Annotating", unit="call")
        iterator = progress

    for dialog, prompt_name, prompt_text, provider, model in iterator:
        dialog_id = dialog.get("dialog_id") or dialog.get("id") or "unknown_dialog_id"

        part_b_input_type, image_path, part_b_text = choose_part_b_input(
            dialog=dialog,
            provider=provider,
            model=model,
            part_b_mode=args.part_b_mode,
        )

        key = f"{CACHE_VERSION}::{args.part_b_mode}::{part_b_input_type}::{dialog_id}::{prompt_name}::{provider}::{model}"
        if key in cache:
            continue

        context = {
            "dialog_id": dialog_id,
            "language": dialog.get("language", "en"),
            "topic_main": dialog.get("topic_main", ""),
            "dialogue_for_annotation": dialog.get("dialogue_for_annotation", ""),
            "part_b_context": part_b_text,
        }

        prompt_contains_placeholders = contains_input_placeholders(prompt_text)
        system_prompt = render_template(prompt_text, context)
        user_text = build_user_text(context, prompt_contains_placeholders)
        messages = make_messages(system_prompt, user_text, image_path)

        result = call_model(
            provider,
            model,
            messages,
            max_retries=args.max_retries,
            retry_base_seconds=args.retry_base_seconds,
            temperature=args.temperature,
        )
        parsed = parse_annotation_response(result.get("response_text"))

        row = {
            "key": key,
            "dialog_id": dialog_id,
            "language": dialog.get("language", "en"),
            "condition_hidden": dialog.get("condition_hidden"),
            "topic_main": dialog.get("topic_main"),
            "prompt_name": prompt_name,
            "provider": provider,
            "model": model,
            "part_b_input_type": part_b_input_type,
            "part_b_image_path": str(image_path.relative_to(REPO_ROOT)) if image_path else None,
            **result,
            **parsed,
        }
        cache[key] = row
        append_cache(CACHE_PATH, row)

        if progress is not None:
            progress.set_postfix_str(
                f"dialog={dialog_id} {provider}/{model} B={part_b_input_type}",
                refresh=False,
            )

        if args.request_pause_seconds > 0:
            time.sleep(args.request_pause_seconds)

    if progress is not None:
        progress.close()

    write_results_csv(cache, OUTPUT_CSV)
    print(f"Saved {len(cache)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
