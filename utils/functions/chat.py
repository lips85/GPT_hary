import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler


class ChatMemory:
    def __init__(self):
        pass

    def save_message(self, message, role):
        st.session_state["messages"].append({"message": message, "role": role})

    # 메시지 전송 함수
    def send_message(self, message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)
        if save:
            self.save_message(message, role)

    # 채팅 기록 표시 함수
    def paint_history(self):
        for message in st.session_state["messages"]:
            self.send_message(message["message"], message["role"], save=False)


# 콜백 핸들러 클래스 정의
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        ChatMemory.save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)
