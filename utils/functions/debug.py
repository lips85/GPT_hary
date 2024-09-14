import streamlit as st
import os


# 디버깅용 지우는 함수
class Debug:
    @staticmethod
    def my_api_key():
        st.session_state["api_key"] = os.environ["OPENAI_API_KEY_PROJECT"]
        st.session_state["api_key_check"] = True

    @staticmethod
    def my_url():
        st.session_state["url"] = os.environ.get(
            "CLAUDEFLARE_SITEMAP_URL", "https://developers.cloudflare.com/sitemap-0.xml"
        )
        st.session_state["url_check"] = True
