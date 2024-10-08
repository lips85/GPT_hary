import re
import streamlit as st
from utils.constant.constant import API_KEY_PATTERN


class SaveEnv:
    @staticmethod
    def save_api_key():
        st.session_state["api_key_check"] = bool(
            re.match(API_KEY_PATTERN, st.session_state["api_key"])
        )

    @staticmethod
    def save_file():
        st.session_state["file_check"] = st.session_state.file is not None

    @staticmethod
    def save_openai_model():
        st.session_state["openai_model_check"] = (
            st.session_state["openai_model"] != "선택해주세요"
        )

    @staticmethod
    def save_url():
        if st.session_state["url"]:
            st.session_state["url_check"] = True
            st.session_state["url_name"] = (
                st.session_state["url"].split("://")[1].replace("/", "_")
            )
        else:
            st.session_state["url_check"] = False
            st.session_state["url_name"] = None
