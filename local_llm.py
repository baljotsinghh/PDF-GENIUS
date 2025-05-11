import requests
import streamlit as st

class LocalLLMAPI:
    def __init__(self):
        # Use Streamlit secrets to configure host, port, and model
        config = st.secrets["local_llm"]
        host = config.get("host", "http://127.0.0.1")
        port = config.get("port", "1234")
        self.base_url = f"{host}:{port}/v1/chat/completions"
        self.model = config.get("model", "llama-3.2-1b-instruct")
        self.temperature = config.get("temperature", 0.7)
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
            return "Error: Could not generate a response from the local model."
