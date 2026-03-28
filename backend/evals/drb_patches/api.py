"""
Provider-aware judge client for DeepResearch Bench evaluation.

This is a DROP-IN REPLACEMENT for the official DRB `utils/api.py`.
It preserves the identical interface (AIClient, call_model, scrape_url)
while supporting configurable judge backends.

DEFAULT BACKEND: DeepSeek (via OpenAI-compatible API)
OPTIONAL BACKEND: Gemini (original DRB default)

IMPORTANT: Using DeepSeek as the judge means results are NOT directly
comparable to official Gemini-judged DRB baselines. This is a custom
evaluation setup using the DRB framework with an alternative judge.

Environment variables:
    DRB_JUDGE_PROVIDER  - "deepseek" (default) or "gemini"
    DEEPSEEK_API_KEY    - Required when provider=deepseek
    DEEPSEEK_BASE_URL   - Optional (default: https://api.deepseek.com)
    DRB_RACE_MODEL      - Model for RACE evaluation (default: deepseek-chat)
    DRB_FACT_MODEL      - Model for FACT extraction/validation (default: deepseek-chat)
    GEMINI_API_KEY      - Required when provider=gemini
    JINA_API_KEY        - Required for FACT web scraping (unchanged)
"""

import os
from typing import Optional, Dict, Any
import requests
import logging


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUDGE_PROVIDER = os.environ.get("DRB_JUDGE_PROVIDER", "deepseek").lower()

# DeepSeek defaults
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_RACE_MODEL = os.environ.get("DRB_RACE_MODEL", "deepseek-chat")
DEEPSEEK_FACT_MODEL = os.environ.get("DRB_FACT_MODEL", "deepseek-chat")

# Gemini defaults (optional fallback)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_RACE_MODEL = os.environ.get("DRB_RACE_MODEL", "gemini-2.5-pro-preview-06-05")
GEMINI_FACT_MODEL = os.environ.get("DRB_FACT_MODEL", "gemini-2.5-flash-preview-05-20")

# Jina (unchanged)
READ_API_KEY = os.environ.get("JINA_API_KEY", "")

# Resolve active models based on provider
if JUDGE_PROVIDER == "deepseek":
    Model = DEEPSEEK_RACE_MODEL
    FACT_Model = DEEPSEEK_FACT_MODEL
elif JUDGE_PROVIDER == "gemini":
    Model = GEMINI_RACE_MODEL
    FACT_Model = GEMINI_FACT_MODEL
else:
    raise ValueError(
        f"Unknown DRB_JUDGE_PROVIDER: '{JUDGE_PROVIDER}'. "
        "Supported: 'deepseek' (default), 'gemini'."
    )

# Legacy compat: API_KEY is the judge API key
API_KEY = DEEPSEEK_API_KEY if JUDGE_PROVIDER == "deepseek" else GEMINI_API_KEY


# ---------------------------------------------------------------------------
# DeepSeek provider (via OpenAI-compatible API)
# ---------------------------------------------------------------------------

class _DeepSeekClient:
    """Judge client using DeepSeek's OpenAI-compatible API."""

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=600.0,
        )
        self.model = model

    def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        model_to_use = model or self.model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                temperature=0.0,
                max_tokens=8192, # Allow for large citation lists
            )
            
            if response.choices[0].finish_reason == "length":
                logger.warning("DeepSeek response was truncated due to output length limit.")
                
            content = response.choices[0].message.content
            if content is None:
                raise Exception("DeepSeek returned empty response content (possibly filtered or refused)")
            return content
        except Exception as e:
            raise Exception(f"DeepSeek generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Gemini provider (original DRB default)
# ---------------------------------------------------------------------------

class _GeminiClient:
    """Judge client using Google Gemini API (original DRB backend)."""

    def __init__(self, api_key: str, model: str):
        from google import genai
        from google.genai import types
        self._types = types

        logging.getLogger('google').setLevel(logging.WARNING)
        logging.getLogger('google.genai').setLevel(logging.WARNING)

        self.client = genai.Client(api_key=api_key, http_options={'timeout': 600000})
        self.model = model

    def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        model_to_use = model or self.model
        contents = [{"role": "user", "parts": [{"text": user_prompt}]}]
        
        # System instructions go in config, not contents for Gemini GenAI SDK
        config = self._types.GenerateContentConfig(
            thinking_config=self._types.ThinkingConfig(thinking_budget=16000)
        )
        if system_prompt:
            config.system_instruction = system_prompt

        try:
            config.max_output_tokens = 8192
            response = self.client.models.generate_content(
                model=model_to_use,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini generation failed: {str(e)}")


# ---------------------------------------------------------------------------
# AIClient — public interface (same as original DRB)
# ---------------------------------------------------------------------------

class AIClient:
    """
    Provider-aware judge client for DRB evaluation.

    Drop-in replacement for the original Gemini-only AIClient.
    Delegates to DeepSeek (default) or Gemini based on DRB_JUDGE_PROVIDER.

    Interface is identical to the original:
        client = AIClient()
        result = client.generate(user_prompt, system_prompt="", model=None)
    """

    def __init__(self, api_key: str = None, model: str = None):
        provider = JUDGE_PROVIDER
        resolved_model = model or Model

        if provider == "deepseek":
            key = api_key or DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise ValueError(
                    "DeepSeek API key not provided. "
                    "Set DEEPSEEK_API_KEY environment variable."
                )
            self._delegate = _DeepSeekClient(
                api_key=key,
                base_url=DEEPSEEK_BASE_URL,
                model=resolved_model,
            )
        elif provider == "gemini":
            key = api_key or GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise ValueError(
                    "Gemini API key not provided. "
                    "Set GEMINI_API_KEY environment variable."
                )
            self._delegate = _GeminiClient(api_key=key, model=resolved_model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        self.model = resolved_model

    def generate(self, user_prompt: str, system_prompt: str = "", model: Optional[str] = None) -> str:
        """Generate text response. Interface matches original DRB AIClient."""
        return self._delegate.generate(user_prompt, system_prompt, model)


# ---------------------------------------------------------------------------
# Jina web scraping — UNCHANGED from original DRB
# ---------------------------------------------------------------------------

class WebScrapingJinaTool:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("Jina API key not provided! Please set JINA_API_KEY environment variable.")

    def __call__(self, url: str) -> Dict[str, Any]:
        try:
            jina_url = f'https://r.jina.ai/{url}'
            headers = {
                "Accept": "application/json",
                'Authorization': self.api_key,
                'X-Timeout': "60000",
                "X-With-Generated-Alt": "true",
            }
            # Set explicit timeout to avoid hanging
            response = requests.get(jina_url, headers=headers, timeout=120)

            if response.status_code != 200:
                raise Exception(f"Jina AI Reader Failed for {url}: {response.status_code}")

            response_dict = response.json()

            return {
                'url': response_dict['data']['url'],
                'title': response_dict['data']['title'],
                'description': response_dict['data']['description'],
                'content': response_dict['data']['content'],
                'publish_time': response_dict['data'].get('publishedTime', 'unknown')
            }

        except Exception as e:
            logger.error(str(e))
            return {
                'url': url,
                'content': '',
                'error': str(e)
            }


_jina_tool = None


def _get_jina_tool() -> WebScrapingJinaTool:
    """Lazy initialize Jina tool to avoid import-time key errors."""
    global _jina_tool
    if _jina_tool is None:
        _jina_tool = WebScrapingJinaTool()
    return _jina_tool


def scrape_url(url: str) -> Dict[str, Any]:
    return _get_jina_tool()(url)


def call_model(user_prompt: str) -> str:
    """FACT helper: uses the FACT model for extraction/validation."""
    client = AIClient(model=FACT_Model)
    return client.generate(user_prompt)


if __name__ == "__main__":
    # Example usage - uncomment with a real URL to test
    # url = "https://example.com"
    # result = scrape_url(url)
    # print(result)
    pass
