import requests
import streamlit as st

class LocalLLMAPI:
    def __init__(self):
        self.host = st.secrets["lm_studio"]["http://192.168.1.38:1234"]   # e.g., "http://your-hostname:port"
        self.model = st.secrets["lm_studio"]["llama-3.2-1b-instruct"] # e.g., "llama-3.2-1b-instruct"
        self.temperature = float(st.secrets["lm_studio"].get("temperature", 0.7))
        self.base_url = f"{self.host}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}

    def generate_response(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature
        }

        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            st.error(f"[LocalLLMAPI ERROR] {e}")
            return "⚠️ Error: Could not generate a response from local model."
