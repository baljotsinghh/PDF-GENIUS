import streamlit as st
import datetime

class ChatMemoryManager:
    def __init__(self, session_state):
        self.session_state = session_state
        if "chat_history" not in self.session_state:
            self.session_state.chat_history = []

    def add_user_message(self, message):
        self.session_state.chat_history.append({"role": "user", "content": message})

    def add_bot_response(self, response):
        self.session_state.chat_history.append({"role": "assistant", "content": response})

    def get_formatted_context(self):
        return "\n".join([f"{c['role'].capitalize()}: {c['content']}" for c in self.session_state.chat_history])

    def get_chat_history(self):
        return self.session_state.chat_history
    
    def add_user_message(self, message):
        self.session_state.chat_history.append({
            "role": "user",
            "content": message,
            "timestamp": str(datetime.datetime.now())
        })

    def add_bot_response(self, response):
        self.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": str(datetime.datetime.now())
        })


    def clear(self):
        self.session_state.chat_history = []
