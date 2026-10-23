import os

# Presets describe where a provider lives, not how to talk to it: every preset
# is reached through the OpenAI-compatible /v1/chat/completions path.
PRESETS = {
    "openai": {
        "base_url": "https://api.openai.com",
        "api_key_env": "OPENAI_API_KEY",
        "default_model": "gpt-4o-mini",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-4-5",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "api_key_env": None,
        "default_model": "llama3.1",
    },
}

DEFAULT_PROVIDER = "openai"

class ProviderConfig:
    def __init__(self, name, base_url, model, api_key=None):
        self.name = name
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

def names():
    return sorted(PRESETS)

def preset(name):
    if name not in PRESETS:
        raise ValueError("unknown provider '%s' (known: %s)" % (name, ", ".join(names())))
    return PRESETS[name]

def resolve(name=None, base_url=None, model=None, api_key=None, env=None):
    """Fill the gaps from the named preset; explicit flags always win."""
    env = os.environ if env is None else env
    name = name or DEFAULT_PROVIDER
    spec = preset(name)
    return ProviderConfig(
        name,
        base_url or spec["base_url"],
        model or spec["default_model"],
        api_key or (env.get(spec["api_key_env"]) if spec["api_key_env"] else None),
    )
