"""
llm_assistant.py — LLM-powered CFO assistant.

Supports multiple FREE providers (no Claude licence required):
  - gemini      : Google Gemini 1.5 Flash — free, fast, capable (★ RECOMMENDED)
  - groq        : Groq Cloud — free API, ultra-fast (Llama 3, Mixtral)
  - mistral     : Mistral AI — free tier (open-mistral-7b)
  - together    : Together AI — free credits (Llama-3-70b)
  - openrouter  : OpenRouter — free models (multiple choices)
  - ollama      : 100% local, no internet, no API key
  - anthropic   : Claude API (paid)
  - openai      : OpenAI API (paid)

All OpenAI-compatible providers share the same _OpenAICompatBackend pattern.
Configure via LLM_PROVIDER and matching key in .env (see .env.example).
"""

import logging
import os
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ─── Provider catalogue (used by UI for status display) ──────────────────────

PROVIDER_INFO: dict[str, dict] = {
    "gemini":     {"label": "Google Gemini",  "free": True,  "url": "https://aistudio.google.com",  "default_model": "gemini-2.0-flash"},
    "groq":       {"label": "Groq Cloud",     "free": True,  "url": "https://console.groq.com",      "default_model": "llama-3.3-70b-versatile"},
    "mistral":    {"label": "Mistral AI",     "free": True,  "url": "https://console.mistral.ai",    "default_model": "open-mistral-7b"},
    "together":   {"label": "Together AI",    "free": True,  "url": "https://api.together.ai",       "default_model": "meta-llama/Llama-3-70b-chat-hf"},
    "openrouter": {"label": "OpenRouter",     "free": True,  "url": "https://openrouter.ai",         "default_model": "meta-llama/llama-3-8b-instruct:free"},
    "ollama":     {"label": "Ollama (local)", "free": True,  "url": "https://ollama.com",            "default_model": "llama3"},
    "anthropic":  {"label": "Anthropic Claude","free": False, "url": "https://console.anthropic.com","default_model": "claude-sonnet-4-6"},
    "openai":     {"label": "OpenAI",         "free": False, "url": "https://platform.openai.com",  "default_model": "gpt-4o-mini"},
}

SYSTEM_PROMPT = """You are an expert CFO assistant and financial advisor with deep knowledge
of corporate finance, FP&A (Financial Planning & Analysis), strategic financial management,
and financial risk assessment — equivalent to a Partner-level expert at a Big 4 advisory firm.

You analyze financial statements, identify trends, explain variances, and provide
boardroom-ready recommendations in clear, executive-level language.

Rules:
- Be concise and data-driven; always cite the specific numbers.
- Flag risks proactively (liquidity, leverage, margin compression, earnings quality).
- Use structured formatting: bullets for lists, **bold** for key figures and conclusions.
- Never hallucinate numbers — only refer to data explicitly provided in the context.
- Structure responses like a senior consultant: situation → complication → recommendation.
- When answering questions, acknowledge uncertainty where data is insufficient.
- Use financial jargon appropriately but explain it when needed for executive clarity."""


def _build_financial_context(financial_data: dict[str, Any]) -> str:
    lines = ["=== FINANCIAL DATA CONTEXT ===\n"]

    if "kpi_summary" in financial_data:
        kpi = financial_data["kpi_summary"]
        lines.append(f"Latest Period: {kpi.get('latest_period', 'N/A')}")
        lines.append("\n-- Latest KPIs --")
        for k, v in kpi.get("latest", {}).items():
            lines.append(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append("\n-- Trailing 4Q Averages --")
        for k, v in kpi.get("trailing_4q_avg", {}).items():
            lines.append(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    if financial_data.get("health_score"):
        hs = financial_data["health_score"]
        lines.append(f"\n-- Financial Health Score --")
        lines.append(f"  Overall: {hs.overall}/100 (Grade: {hs.grade}) — {hs.assessment}")
        lines.append(f"  Profitability: {hs.profitability_score}/25")
        lines.append(f"  Liquidity: {hs.liquidity_score}/20")
        lines.append(f"  Leverage: {hs.leverage_score}/20")
        lines.append(f"  Growth: {hs.growth_score}/20")
        lines.append(f"  Cash Quality: {hs.cash_quality_score}/15")

    if financial_data.get("anomalies"):
        lines.append("\n-- Detected Anomalies --")
        for a in financial_data["anomalies"]:
            lines.append(f"  [{a['period']}] {a['metric']}: {a['description']}")

    if "forecast_summary" in financial_data:
        lines.append("\n-- Forecasts (next 4 quarters) --")
        fc = financial_data["forecast_summary"]
        if isinstance(fc, pd.DataFrame):
            lines.append(fc.to_string(index=False))

    if "raw_income" in financial_data:
        lines.append("\n-- Income Statement (last 8 quarters) --")
        lines.append(financial_data["raw_income"].tail(8).to_string())

    if "raw_balance" in financial_data:
        lines.append("\n-- Balance Sheet (last 4 quarters) --")
        lines.append(financial_data["raw_balance"].tail(4).to_string())

    return "\n".join(lines)


# ─── Provider backends ────────────────────────────────────────────────────────

class _OpenAICompatBackend:
    """Generic OpenAI-compatible backend. Subclasses set _ENV_KEY and _BASE_URL."""

    _ENV_KEY: str = ""
    _BASE_URL: str = ""
    _SIGNUP_URL: str = ""
    DEFAULT_MODEL: str = ""

    def __init__(self):
        from openai import OpenAI
        api_key = os.environ.get(self._ENV_KEY, "")
        if not api_key:
            raise EnvironmentError(
                f"{self._ENV_KEY} not set. Get a free key at {self._SIGNUP_URL}"
            )
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self._BASE_URL:
            kwargs["base_url"] = self._BASE_URL
        if hasattr(self, "_extra_headers"):
            kwargs["default_headers"] = self._extra_headers
        self.client = OpenAI(**kwargs)
        self.model = os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)

    def chat(self, messages: list[dict], max_tokens: int = 1200) -> str:
        import time
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                return response.choices[0].message.content
            except Exception as e:
                last_err = e
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
                    wait = 30 * (attempt + 1)
                    logger.warning("Rate limit (attempt %d/3) — waiting %ds…", attempt + 1, wait)
                    time.sleep(wait)
                else:
                    raise
        raise last_err  # type: ignore[misc]


class _GeminiBackend(_OpenAICompatBackend):
    """Google Gemini — free API via OpenAI-compatible endpoint.
    Get free key: https://aistudio.google.com → Get API key
    Recommended: gemini-2.0-flash (free, latest, excellent quality)
    """
    _ENV_KEY = "GEMINI_API_KEY"
    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    _SIGNUP_URL = "https://aistudio.google.com"
    DEFAULT_MODEL = "gemini-2.0-flash"


class _GroqBackend(_OpenAICompatBackend):
    """Groq Cloud — free API, ultra-fast inference.
    Get free key: https://console.groq.com → API Keys
    Models: llama-3.3-70b-versatile | llama-3.1-8b-instant | mixtral-8x7b-32768
    """
    _ENV_KEY = "GROQ_API_KEY"
    _BASE_URL = "https://api.groq.com/openai/v1"
    _SIGNUP_URL = "https://console.groq.com"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"


class _MistralBackend(_OpenAICompatBackend):
    """Mistral AI — free tier available.
    Get free key: https://console.mistral.ai → API Keys
    Free models: open-mistral-7b | open-mixtral-8x7b
    """
    _ENV_KEY = "MISTRAL_API_KEY"
    _BASE_URL = "https://api.mistral.ai/v1"
    _SIGNUP_URL = "https://console.mistral.ai"
    DEFAULT_MODEL = "open-mistral-7b"


class _TogetherBackend(_OpenAICompatBackend):
    """Together AI — free credits on signup.
    Get key: https://api.together.ai → API Keys
    Models: meta-llama/Llama-3-70b-chat-hf | mistralai/Mixtral-8x7B-Instruct-v0.1
    """
    _ENV_KEY = "TOGETHER_API_KEY"
    _BASE_URL = "https://api.together.xyz/v1"
    _SIGNUP_URL = "https://api.together.ai"
    DEFAULT_MODEL = "meta-llama/Llama-3-70b-chat-hf"


class _OpenRouterBackend(_OpenAICompatBackend):
    """OpenRouter — access many free models via one API.
    Get free key: https://openrouter.ai → Keys
    Free models: meta-llama/llama-3-8b-instruct:free | google/gemma-2-9b-it:free
    """
    _ENV_KEY = "OPENROUTER_API_KEY"
    _BASE_URL = "https://openrouter.ai/api/v1"
    _SIGNUP_URL = "https://openrouter.ai"
    DEFAULT_MODEL = "meta-llama/llama-3-8b-instruct:free"
    _extra_headers = {
        "HTTP-Referer": "https://github.com/cfo-copilot",
        "X-Title": "CFO Copilot",
    }


class _OllamaBackend(_OpenAICompatBackend):
    """100% local, no API key, no internet.
    Install: https://ollama.com then run: ollama pull llama3
    """
    _ENV_KEY = ""   # no key needed
    _SIGNUP_URL = "https://ollama.com"
    DEFAULT_MODEL = "llama3"

    def __init__(self):
        from openai import OpenAI
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        self.model = os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)


class _AnthropicBackend:
    """Anthropic Claude — paid API."""

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self):
        import anthropic as _anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY not set.")
        self.client = _anthropic.Anthropic(api_key=api_key)
        self.model = os.environ.get("LLM_MODEL", self.DEFAULT_MODEL)

    def chat(self, messages: list[dict], max_tokens: int = 1200) -> str:
        # Remove system messages from array (Anthropic uses separate system param)
        user_messages = [m for m in messages if m["role"] != "system"]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=user_messages,
        )
        return response.content[0].text


def _get_backend(provider: str | None = None):
    provider = (provider or os.environ.get("LLM_PROVIDER", "gemini")).lower()
    backends = {
        "gemini":     _GeminiBackend,
        "groq":       _GroqBackend,
        "mistral":    _MistralBackend,
        "together":   _TogetherBackend,
        "openrouter": _OpenRouterBackend,
        "ollama":     _OllamaBackend,
        "anthropic":  _AnthropicBackend,
        "openai":     lambda: _OpenAICompatBackend.__class__.__new__(_OpenAICompatBackend),  # handled below
    }
    if provider == "openai":
        class _OpenAIBackend(_OpenAICompatBackend):
            _ENV_KEY = "OPENAI_API_KEY"
            _BASE_URL = ""
            _SIGNUP_URL = "https://platform.openai.com"
            DEFAULT_MODEL = "gpt-4o-mini"
        return _OpenAIBackend()

    if provider not in backends:
        raise ValueError(f"Unknown LLM_PROVIDER '{provider}'. Choose: {list(backends)}")
    logger.info("LLM provider: %s", provider)
    return backends[provider]()


def get_provider_status() -> dict[str, bool]:
    """Return dict of provider_name → True if API key is configured."""
    env_keys = {
        "gemini":     "GEMINI_API_KEY",
        "groq":       "GROQ_API_KEY",
        "mistral":    "MISTRAL_API_KEY",
        "together":   "TOGETHER_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
        "openai":     "OPENAI_API_KEY",
        "ollama":     None,
    }
    return {
        provider: (key is None or bool(os.environ.get(key, "")))
        for provider, key in env_keys.items()
    }


# ─── Public CFOAssistant ──────────────────────────────────────────────────────

class CFOAssistant:
    """Provider-agnostic CFO assistant with automatic fallback on rate limits."""

    # Free providers tried in order when primary hits rate limit
    _FALLBACK_ORDER = ["groq", "together", "openrouter", "mistral", "ollama"]

    def __init__(self, provider: str | None = None):
        self._backend = _get_backend(provider)
        self._conversation_history: list[dict] = []
        self.provider = provider or os.environ.get("LLM_PROVIDER", "groq")

    def _messages_with_system(self, messages: list[dict]) -> list[dict]:
        return [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    def _chat(self, messages: list[dict], max_tokens: int) -> str:
        """Chat with automatic fallback to other free providers on rate limit."""
        try:
            return self._backend.chat(messages, max_tokens)
        except Exception as e:
            err = str(e)
            if not ("429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower()):
                raise
            # Rate limited — try configured free providers as fallback
            logger.warning("Rate limit on %s — trying fallbacks…", self.provider)
            status = get_provider_status()
            for fb in self._FALLBACK_ORDER:
                if fb == self.provider or not status.get(fb, False):
                    continue
                try:
                    result = _get_backend(fb).chat(messages, max_tokens)
                    logger.info("Fallback to %s succeeded.", fb)
                    return result
                except Exception as fb_err:
                    logger.warning("Fallback %s failed: %s", fb, fb_err)
            raise RuntimeError(
                f"All providers exhausted. Primary error: {err}\n"
                "Configure GROQ_API_KEY at https://console.groq.com (free, generous limits)."
            ) from e

    def generate_executive_summary(self, financial_data: dict[str, Any]) -> str:
        context = _build_financial_context(financial_data)
        company = financial_data.get("company_name", "the company")
        messages = self._messages_with_system([
            {"role": "user", "content": (
                f"{context}\n\n"
                f"Generate a comprehensive executive financial summary for {company} covering:\n"
                "1. **Executive Summary** — 2-sentence headline of financial performance\n"
                "2. **Financial Performance** — revenue, margins, profitability vs prior periods\n"
                "3. **Cash Flow & Liquidity** — OCF, working capital, current ratio assessment\n"
                "4. **Leverage & Debt Profile** — D/E ratio, interest coverage, debt sustainability\n"
                "5. **Financial Health Score Analysis** — interpretation of the composite score\n"
                "6. **Key Risks & Anomalies** — ranked by severity with quantified impact\n"
                "7. **Outlook & Forecast** — projected trends next 4 quarters with confidence\n"
                "8. **Strategic Recommendations** — 3-5 specific, actionable items for the CFO\n\n"
                "Format this for a Board/CFO audience. Be direct, use numbers, be advisory."
            )}
        ])
        return self._chat(messages, max_tokens=1800)

    def generate_scenario_narrative(
        self, scenario_name: str, base: dict, scenario: dict, assumptions: dict
    ) -> str:
        delta_revenue = (scenario["revenue"] - base["revenue"]) / base["revenue"] * 100
        delta_ebitda = (scenario["ebitda"] - base["ebitda"]) / base["ebitda"] * 100
        delta_net = (scenario["net_income"] - base["net_income"]) / base["net_income"] * 100
        messages = self._messages_with_system([
            {"role": "user", "content": (
                f"Scenario: **{scenario_name}**\n"
                f"Assumptions applied:\n"
                + "\n".join(f"  - {k}: {v}" for k, v in assumptions.items()) + "\n\n"
                f"Impact summary:\n"
                f"  - Revenue: ${base['revenue']/1e6:.2f}M → ${scenario['revenue']/1e6:.2f}M ({delta_revenue:+.1f}%)\n"
                f"  - EBITDA: ${base['ebitda']/1e6:.2f}M → ${scenario['ebitda']/1e6:.2f}M ({delta_ebitda:+.1f}%)\n"
                f"  - Net Income: ${base['net_income']/1e6:.2f}M → ${scenario['net_income']/1e6:.2f}M ({delta_net:+.1f}%)\n"
                f"  - EBITDA Margin: {base['ebitda_margin']*100:.1f}% → {scenario['ebitda_margin']*100:.1f}%\n\n"
                "In 4-6 sentences: (1) Explain what this scenario represents strategically, "
                "(2) assess the severity of the impact, (3) identify the primary financial risk, "
                "and (4) give one specific mitigation action the CFO should consider."
            )}
        ])
        return self._chat(messages, max_tokens=500)

    def explain_kpi_variance(
        self, metric: str, current_value: float, prior_value: float,
        financial_data: dict[str, Any],
    ) -> str:
        pct = ((current_value - prior_value) / abs(prior_value)) * 100 if prior_value else 0
        direction = "increased" if pct > 0 else "decreased"
        context = _build_financial_context(financial_data)
        messages = self._messages_with_system([
            {"role": "user", "content": (
                f"{context}\n\n"
                f"The **{metric}** has {direction} from {prior_value:.3f} "
                f"to {current_value:.3f} ({pct:+.1f}%).\n\n"
                "Explain in 3-4 sentences: what likely drove this change, "
                "whether it is concerning in context, and what the CFO should monitor."
            )}
        ])
        return self._chat(messages, max_tokens=500)

    def answer_question(self, question: str, financial_data: dict[str, Any]) -> str:
        if not self._conversation_history:
            context = _build_financial_context(financial_data)
            self._conversation_history.append({
                "role": "user",
                "content": f"{context}\n\nI have loaded the financial data above. I'll ask questions about it."
            })
            self._conversation_history.append({
                "role": "assistant",
                "content": "Understood. I have reviewed the financial statements, KPIs, health score, and forecasts. Please go ahead with your questions."
            })

        self._conversation_history.append({"role": "user", "content": question})
        messages = self._messages_with_system(self._conversation_history)
        answer = self._chat(messages, max_tokens=800)
        self._conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def reset_conversation(self) -> None:
        self._conversation_history = []
