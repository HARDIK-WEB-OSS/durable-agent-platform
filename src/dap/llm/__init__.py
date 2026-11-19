import os

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
# Stop LiteLLM's import-time load_dotenv(), which copies .env (API keys included) into
# os.environ. Settings already reads .env; nothing else should see it.
os.environ.setdefault("LITELLM_MODE", "PRODUCTION")
