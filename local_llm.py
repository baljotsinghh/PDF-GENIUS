# components/LocalLLMAPI.py

import requests
import streamlit as st


class LocalLLMAPI:
    def __init__(self, host= st.secrets("LOCAL_LLM_SERVER"), model= st.secrets("MODEL_API_IDENTIFIER"), temperature=0.7):
        self.base_url = f"{host}/v1/chat/completions"
        self.model = model
        self.temperature = temperature
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
            print(f"[LocalLLMAPI ERROR] {e}")
            return "Error: Could not generate a response from local model."
