import json
import logging
import re
from typing import Any

import httpx

from config.settings import get_settings

logger = logging.getLogger(__name__)

OLLAMA_TIMEOUT = 30.0
HF_TIMEOUT = 60.0


async def _call_ollama(prompt: str, system_prompt: str) -> str:
    settings = get_settings()
    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        response = await client.post(url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned status {response.status_code}: {response.text}")
        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "")
        if not content:
            raise RuntimeError("Ollama returned empty response")
        return content


# async def _call_huggingface(prompt: str, system_prompt: str) -> str:
#     settings = get_settings()
#     print(f"Using HuggingFace model: {settings.HUGGINGFACE_MODEL}"  )
#     if not settings.HUGGINGFACE_API_KEY:
#         raise RuntimeError("HUGGINGFACE_API_KEY is not configured")

#     url = f"https://api-inference.huggingface.co/models/{settings.HUGGINGFACE_MODEL}"
#     headers = {
#         "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
#         "Content-Type": "application/json",
#     }
#     full_prompt = f"<s>[INST] {system_prompt}\n\n{prompt} [/INST]"
#     payload = {
#         "inputs": full_prompt,
#         "parameters": {
#             "max_new_tokens": 2048,
#             "return_full_text": False,
#         },
#     }
#     async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
#         response = await client.post(url, headers=headers, json=payload)
#         if response.status_code != 200:
#             raise RuntimeError(
#                 f"HuggingFace returned status {response.status_code}: {response.text}"
#             )
#         data = response.json()
#         if isinstance(data, list) and len(data) > 0:
#             generated = data[0].get("generated_text", "")
#             if generated:
#                 return generated
#         if isinstance(data, dict) and "generated_text" in data:
#             return data["generated_text"]
#         raise RuntimeError(f"HuggingFace returned unexpected format: {data}")

async def _call_huggingface(prompt: str, system_prompt: str) -> str:
    settings = get_settings()
    if not settings.HUGGINGFACE_API_KEY:
        raise RuntimeError("HUGGINGFACE_API_KEY is not configured")

    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.HUGGINGFACE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.HUGGINGFACE_MODEL, 
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
    }

    async with httpx.AsyncClient(timeout=HF_TIMEOUT) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(
                f"HuggingFace returned status {response.status_code}: {response.text}"
            )
        data = response.json()
        return data["choices"][0]["message"]["content"]
def parse_llm_json(text: str, default: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        logger.warning("Failed to parse LLM JSON, using default structure")
        return default


async def get_llm_response(prompt: str, system_prompt: str) -> str:
    try:
        return await _call_ollama(prompt, system_prompt)
    except Exception as exc:
        logger.warning("Ollama unavailable, falling back to HuggingFace: %s", exc)
        try:
            return await _call_huggingface(prompt, system_prompt)
        except Exception as hf_exc:
            raise RuntimeError(
                f"Both LLM providers failed. Ollama: {exc}. HuggingFace: {hf_exc}"
            ) from hf_exc
