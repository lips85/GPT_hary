import json
import streamlit as st
import openai
from utils.constant.constant import OPENAI_MODEL
from utils.functions.chat import ChatMemory
from utils.functions.save_env import SaveEnv
from dotenv import load_dotenv
from utils.functions.debug import Debug

load_dotenv()

# SaveEnv 인스턴스 생성
save_env = SaveEnv()

# 세션 상태 초기화
for key, default in [
    ("messages", []),
    ("api_key", None),
    ("api_key_check", False),
    ("openai_model", "선택해주세요"),
    ("openai_model_check", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Streamlit 페이지 설정
st.set_page_config(
    page_title="AssistantGPT",
    page_icon="🚀",
    layout="wide",
)

# 페이지 제목 및 설명
st.title("🚀 리서치 마스터 🚀")

st.markdown(
    """
    검색은 저에게 맡겨주세요! 여러분들의 시간을 아껴드리겠습니다.
    (OpenAI Assistant API 사용)
 """
)


# OpenAI Chat Completion API를 사용하여 메시지 전송 및 응답 처리
def generate_response(api_key, model, messages):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model, messages=messages, max_tokens=150
    )
    return response["choices"][0]["message"]["content"]


# UI 및 API key 입력 부분 설정
with st.sidebar:
    # API Key 입력 필드
    st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        on_change=save_env.save_api_key,  # 인스턴스 메서드로 변경
        key="api_key",
        type="password",
    )

    if st.session_state["api_key_check"]:
        st.success("😄 API_KEY가 유효합니다.")
    else:
        st.warning("API_KEY를 입력하세요.")

    st.button("디버깅용 버튼", on_click=Debug.my_api_key)

    # OpenAI 모델 선택 박스
    st.selectbox(
        "OpenAI Model을 선택하세요.",
        options=OPENAI_MODEL,
        on_change=save_env.save_openai_model,  # 인스턴스 메서드로 변경
        key="openai_model",
    )

    if st.session_state["openai_model_check"]:
        st.success("😄 모델이 선택되었습니다.")
    else:
        st.warning("모델을 선택해주세요.")

    st.write(
        """
        Made by hary.
        
        Github
        https://github.com/lips85/GPT_hary

        streamlit
        https://hary-gpt.streamlit.app/
        """
    )

# API Key와 Model 설정이 완료되지 않았을 때 경고 메시지
if not (st.session_state["api_key_check"] and st.session_state["openai_model_check"]):

    if not st.session_state["api_key_check"]:
        st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

    if not st.session_state["openai_model_check"]:
        st.warning("Please write down a **:blue[OpenAI Model Select]** on the sidebar.")

else:
    # Chat Memory 객체 생성
    discussion_client = ChatMemory()

    # 이전 대화 기록 출력 (중복 방지)
    discussion_client.paint_history()

    # 사용자 입력 받기
    query = st.chat_input("Ask a question to the assistant.")
    if query:
        # 메시지 저장 및 출력
        discussion_client.send_message(query, "user")

        # OpenAI API에 메시지 전달
        messages = [{"role": "system", "content": "You are a helpful assistant."}]

        # 이전 메시지를 추가
        for message in st.session_state["messages"]:
            messages.append({"role": message["role"], "content": message["message"]})

        # OpenAI API 호출
        response = generate_response(
            st.session_state["api_key"], st.session_state["openai_model"], messages
        )

        # AI 응답 저장 및 출력
        discussion_client.send_message(response, "assistant")

        # 채팅 내역 다운로드 버튼
        st.download_button(
            label="채팅 내역 다운로드",
            data=json.dumps(st.session_state["messages"]),
            file_name="chat_history.txt",
            mime="text/plain",
        )
