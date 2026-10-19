import os
import json
import urllib.request
import urllib.error

class Provider:
    def __init__(self, name, base_url, model, api_key=None):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("EVALS_API_KEY")

    def chat(self, prompt):
        url = self.base_url + "/v1/chat/completions"
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", "Bearer " + self.api_key)
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
        obj = json.loads(raw)
        choice = obj["choices"][0]["message"]["content"]
        usage = obj.get("usage", {}) or {}
        return {
            "output": choice,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }

class MockProvider(Provider):
    def __init__(self, model="mock"):
        super().__init__("mock", "http://localhost", model, api_key="mock")
    def chat(self, prompt):
        return {"output": "mock response: " + prompt[:80], "prompt_tokens": 10, "completion_tokens": 5}
