import streamlit as st
import os


# 디버깅용 지우는 함수
class Debug:
    def __init__(self):
        pass

    def my_api_key(self):
        st.session_state["api_key"] = os.environ["OPENAI_API_KEY_PROJECT"]
        st.session_state["api_key_check"] = True

    def my_url(self):
        st.session_state["url"] = os.environ.get(
            "CLAUDEFLARE_SITEMAP_URL", "https://developers.cloudflare.com/sitemap-0.xml"
        )
        st.session_state["url_check"] = True
