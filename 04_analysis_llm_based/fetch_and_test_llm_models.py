"""
1. Fetch available models from NHR and GWDG providers via their /models endpoints.
2. Test connectivity for every fetched model.
3. Test the OpenAI key directly with the model configured in .env (OPENAI_MODEL).
4. Save the models that passed connectivity testing to configs/llm_apis_ready.json.

Run from the repo root:  python 04_analysis_llm_based/fetch_and_test_llm_models.py
"""

from openai import OpenAI
from dotenv import load_dotenv
import os, json, time, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

with (REPO_ROOT / "configs" / "paths.json").open("r", encoding="utf-8") as _f:
    PATHS = {k: REPO_ROOT / v for k, v in json.load(_f).items() if not k.startswith("_")}

load_dotenv(dotenv_path=PATHS["env_file"])

READY_MODELS_PATH = PATHS["llm_apis_ready"]

PROVIDERS = {
    "nhr": {
        "api_key": os.getenv("NHR_API_KEY"),
        "base_url": os.getenv("NHR_BASE_URL"),
    },
    "gwdg": {
        "api_key": os.getenv("GWDG_API_KEY"),
        "base_url": os.getenv("GWDG_BASE_URL"),
    },
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

# Model ID substrings that identify non-chat models to skip
EXCLUDE_KEYWORDS = {"embed", "bge-", "e5-", "ocr", "rerank", "whisper", "clip"}

TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Reply with exactly one sentence confirming you are working."},
]


def _client(provider_name: str) -> OpenAI:
    cfg = PROVIDERS[provider_name]
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"], timeout=30.0)


def is_chat_model(model_id: str) -> bool:
    mid = model_id.lower()
    return not any(kw in mid for kw in EXCLUDE_KEYWORDS)


def fetch_models(provider_name: str) -> dict[str, str]:
    """Return {model_id: model_id} for all chat-capable models at this provider."""
    client = _client(provider_name)
    models: dict[str, str] = {}
    try:
        for model in client.models.list().data:
            if is_chat_model(model.id):
                models[model.id] = model.id
        print(f"  {provider_name.upper()}: {len(models)} chat models found")
    except Exception as e:
        print(f"  {provider_name.upper()}: ERROR fetching models — {e}")
    return models


def save_ready_models(results: list[dict]) -> None:
    """Save only the models that passed their connectivity test, grouped by provider."""
    today = str(datetime.date.today())
    output: dict = {"_comment": f"Models that passed connectivity testing — last checked {today}."}
    for r in results:
        if r["status"] != "OK":
            continue
        output.setdefault(r["provider"], {})[r["model_id"]] = r["model_id"]

    READY_MODELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(READY_MODELS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {READY_MODELS_PATH}\n")


def test_model(provider_name: str, model_id: str) -> dict:
    client = _client(provider_name)
    result = {
        "provider": provider_name,
        "model_id": model_id,
        "status": None,
        "response": None,
        "error": None,
        "latency_s": None,
    }
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=TEST_MESSAGES,
            temperature=0.1,
            max_tokens=64,
        )
        result["latency_s"] = round(time.time() - t0, 2)
        content = resp.choices[0].message.content
        result["response"] = content.strip() if content else ""
        result["status"] = "OK"
    except Exception as e:
        result["latency_s"] = round(time.time() - t0, 2)
        result["status"] = "FAIL"
        result["error"] = str(e)
    return result


def test_openai() -> dict:
    """OpenAI's catalog has hundreds of non-chat models, so unlike NHR/GWDG we
    don't fetch+test the whole catalog — just confirm the configured key/model work."""
    result = {
        "provider": "openai",
        "model_id": OPENAI_MODEL,
        "status": None,
        "response": None,
        "error": None,
        "latency_s": None,
    }
    if not OPENAI_API_KEY:
        result["status"] = "FAIL"
        result["error"] = "OPENAI_API_KEY is not set."
        result["latency_s"] = 0.0
        return result

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=30.0)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=TEST_MESSAGES,
            temperature=0.1,
            max_completion_tokens=64,
        )
        result["latency_s"] = round(time.time() - t0, 2)
        content = resp.choices[0].message.content
        result["response"] = content.strip() if content else ""
        result["status"] = "OK"
    except Exception as e:
        result["latency_s"] = round(time.time() - t0, 2)
        result["status"] = "FAIL"
        result["error"] = str(e)
    return result


def main() -> None:
    # ── Step 1: fetch available models from providers ──────────────────────────
    print("=" * 60)
    print("Fetching available models from providers...")
    print("=" * 60)
    api_models: dict[str, dict[str, str]] = {}
    for provider_name in PROVIDERS:
        api_models[provider_name] = fetch_models(provider_name)

    total = sum(len(v) for v in api_models.values())
    print(f"\nTesting {total} models across {len(api_models)} providers\n")

    # ── Step 2: test connectivity ──────────────────────────────────────────────
    results: list[dict] = []
    for provider_name, models in api_models.items():
        print(f"{'=' * 60}")
        print(f"Provider: {provider_name.upper()}  |  {PROVIDERS[provider_name]['base_url']}")
        print(f"{'=' * 60}")
        for model_id in models:
            print(f"  {model_id} ...", end=" ", flush=True)
            r = test_model(provider_name, model_id)
            results.append(r)
            if r["status"] == "OK":
                print(f"OK ({r['latency_s']}s)")
                print(f"    {r['response'][:120]}")
            else:
                print(f"FAIL ({r['latency_s']}s)")
                print(f"    {r['error'][:200]}")

    # ── Step 3: test the OpenAI key directly ───────────────────────────────────
    print(f"{'=' * 60}")
    print("Provider: OPENAI  |  api.openai.com")
    print(f"{'=' * 60}")
    print(f"  {OPENAI_MODEL} ...", end=" ", flush=True)
    openai_result = test_openai()
    results.append(openai_result)
    if openai_result["status"] == "OK":
        print(f"OK ({openai_result['latency_s']}s)")
        print(f"    {openai_result['response'][:120]}")
    else:
        print(f"FAIL ({openai_result['latency_s']}s)")
        print(f"    {openai_result['error'][:200]}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"Summary: {ok}/{len(results)} models OK\n")
    for r in results:
        mark = "OK  " if r["status"] == "OK" else "FAIL"
        print(f"  {mark}  [{r['provider']}] {r['model_id']:<55} {r['latency_s']}s")

    save_ready_models(results)


if __name__ == "__main__":
    main()
