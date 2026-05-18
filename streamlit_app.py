"""
streamlit_app.py — Entry point for Streamlit Cloud deployment.

Loads API keys from st.secrets (Streamlit Cloud) or from .env (local).
Run locally:  streamlit run streamlit_app.py
"""

import os
import streamlit as st

# ── Inject Streamlit Cloud secrets into environment variables ─────────────────
# On Streamlit Cloud, API keys are stored in the Secrets dashboard.
# This makes them available to os.environ so llm_assistant.py can read them.
_SECRET_KEYS = [
    "LLM_PROVIDER", "LLM_MODEL",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_BASE_URL",
]
for _key in _SECRET_KEYS:
    try:
        if _key in st.secrets and not os.environ.get(_key):
            os.environ[_key] = st.secrets[_key]
    except Exception:
        pass

# ── Launch dashboard ──────────────────────────────────────────────────────────
from src.dashboard import main
main()
