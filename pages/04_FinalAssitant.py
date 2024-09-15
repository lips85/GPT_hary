import time
import json
import streamlit as st
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.document_loaders.web_base import WebBaseLoader
from openai import OpenAI

# 환경변수 저장, 상수 임포트, 채팅 클래스
from utils.functions.chat import ChatMemory
from utils.constant.constant import OPENAI_MODEL
from utils.functions.save_env import SaveEnv


# 디버깅용 임포트 (업로드시 주석처리)
# from utils.functions.debug import Debug
# from dotenv import load_dotenv
# load_dotenv()


# 세션 상태 초기화
for key, default in [
    ("messages", []),
    ("api_key", None),
    ("api_key_check", False),
    ("openai_model", "선택해주세요"),
    ("openai_model_check", False),
    ("query", None),
    ("assistant", None),
    ("thread", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.set_page_config(
    page_title="AssistantGPT",
    page_icon="🚀",
    layout="wide",
)
st.title("🚀 리서치 마스터 🚀")

if not (st.session_state["api_key_check"] and st.session_state["openai_model_check"]):

    st.markdown(
        """
        검색은 저에게 맡겨주세요! 여러분들의 시간을 아껴드리겠습니다.
        (OpenAI Assistant API 사용)
    """
    )


# openai assistant 클래스
class ThreadClient:
    def __init__(self, client):
        self.client = client

    def create_run(self, assistant_id, thread_id):
        run = self.client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        return run

    def get_run(self, run_id, thread_id):
        return self.client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )

    def send_message(self, thread_id, content):
        return self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=content,
        )

    def get_messages(self, thread_id):
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        messages = list(messages)
        messages.reverse()
        return messages

    def get_tool_outputs(self, run_id, thread_id):
        run = self.get_run(run_id, thread_id)
        outputs = []
        for action in run.required_action.submit_tool_outputs.tool_calls:
            action_id = action.id
            function = action.function
            outputs.append(
                {
                    "output": functions_map[function.name](
                        json.loads(function.arguments)
                    ),
                    "tool_call_id": action_id,
                }
            )
        return outputs

    def submit_tool_outputs(self, run_id, thread_id):
        outputs = self.get_tool_outputs(run_id, thread_id)
        return self.client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id,
            thread_id=thread_id,
            tool_outputs=outputs,
        )


# 위키피디아, 덕덕고 검색 클래스
class IssueSearchClient:

    def __init__(self):
        self.ddg = DuckDuckGoSearchAPIWrapper()
        self.w = WikipediaAPIWrapper()
        self.loader = WebBaseLoader()

    def get_websites_by_wikipedia_search(self, inputs):
        self.w = WikipediaAPIWrapper()
        query = inputs["query"]
        return self.w.run(query)

    def get_websites_by_duckduckgo_search(self, inputs):
        self.ddg = DuckDuckGoSearchAPIWrapper()
        query = inputs["query"]
        return self.ddg.run(query)

    def get_document_text(self, inputs):
        url = inputs["url"]
        self.loader = WebBaseLoader([url])
        self.docs = self.loader.load()
        return self.docs[0].page_content

    def save_file(self, inputs):
        text = inputs["text"]
        with open("research_report.txt", "w", encoding="utf-8") as f:
            f.write(text)
        st.download_button(label="다운로드", file_name="research_report.txt", data=text)
        return "저장 완료"


issue_search_client = IssueSearchClient()
discussion_client = ChatMemory()

functions_map = {
    "get_websites_by_wikipedia_search": issue_search_client.get_websites_by_wikipedia_search,
    "get_websites_by_duckduckgo_search": issue_search_client.get_websites_by_duckduckgo_search,
    "get_document_text": issue_search_client.get_document_text,
    "save_file": issue_search_client.save_file,
}

# 도구 설명을 한국어로 변경
functions = [
    {
        "type": "function",
        "function": {
            "name": "get_websites_by_wikipedia_search",
            "description": "주어진 쿼리에 대한 웹사이트를 찾기 위해 이 도구를 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 쿼리입니다. 예: XZ 백도어에 대한 연구",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_websites_by_duckduckgo_search",
            "description": "주어진 쿼리에 대한 웹사이트를 찾기 위해 이 도구를 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "검색할 쿼리입니다. 예: XZ 백도어에 대한 연구",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document_text",
            "description": "주어진 URL의 웹사이트를 로드하기 위해 이 도구를 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "로드할 URL입니다. 예: https://ko.wikipedia.org/wiki/백도어",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_file",
            "description": "텍스트를 파일로 저장하기 위해 이 도구를 사용하세요.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "저장할 텍스트입니다.",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


def get_assistant(client):
    assistant = client.beta.assistants.create(
        name="리서치 어시스턴트",
        instructions=(
            "당신은 fuctions를 이용하여 사용자가 원하는 키워드를 검색, 요약, 저장하는데 도움이 되는 Assistant 입니다."
            "모든 정보들은 markdown 형식으로 작성하세요."
            "최대한 많은 정보를 자세한 내용으로 제공하세요."
            "각각의 자료 출처들을 반드시 표기하세요."
            "모든 응답은 한국어로 작성하세요."
            "최종 답변은 모든 출처와 관련 링크를 포함해 변경없이 동일하게 .txt 파일에 저장해야 합니다."
        ),
        model="gpt-4o-mini-2024-07-18",
        temperature=0.1,
        tools=functions,
    )
    return assistant


with st.sidebar:
    # API Key 입력 필드
    st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        on_change=SaveEnv.save_api_key,
        key="api_key",
        type="password",
    )

    if st.session_state["api_key_check"]:
        st.success("😄 API_KEY가 유효합니다.")
    else:
        st.warning("API_KEY를 입력하세요.")

    # 디버깅용 버튼 (업로드시 주석처리)
    # st.button("디버깅용 버튼", on_click=Debug.my_api_key)

    # OpenAI 모델 선택 박스
    st.selectbox(
        "OpenAI Model을 선택하세요.",
        options=OPENAI_MODEL,
        on_change=SaveEnv.save_openai_model,
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

if not (st.session_state["api_key_check"] and st.session_state["openai_model_check"]):

    if not st.session_state["api_key_check"]:
        st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

    if not st.session_state["openai_model_check"]:
        st.warning("Please write down a **:blue[OpenAI Model Select]** on the sidebar.")

else:
    st.chat_message("ai").markdown(
        "안녕하세요. 저는 리서치 어시스턴트입니다. 무엇을 도와드릴까요?"
    )
    client = OpenAI(api_key=st.session_state["api_key"])

    if st.session_state["assistant"] is None:
        st.session_state["assistant"] = get_assistant(client)
        assistant_id = st.session_state["assistant"].id
    else:
        assistant_id = st.session_state["assistant"].id

    # 이전 메시지 히스토리 표시
    if st.session_state["messages"]:
        discussion_client.paint_history()

    query = st.chat_input("웹사이트에 질문을 해보세요.")

    if query:
        # 새로운 메시지를 세션 상태에 추가
        discussion_client.save_message(query, "human")

        # 사용자 메시지 표시
        st.chat_message("human").markdown(query)

        # 답변이 표시될 공간을 미리 할당
        response_placeholder = st.empty()

        # 스피너를 사용하는 동안 이전 답변이 복사되지 않도록 placeholder에 '답변 생성 중...' 텍스트 표시
        with st.spinner(f"🔍 :blue[{query}] 답변 생성 중.. "):
            thread = client.beta.threads.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"{query}",
                    }
                ]
            )
            thread_id = thread.id
            assistant_client = ThreadClient(client)
            run = assistant_client.create_run(assistant_id, thread_id)
            run_id = run.id

            # 답변 대기 중에도 기존 메시지를 계속 표시
            while assistant_client.get_run(run_id, thread_id).status in [
                "queued",
                "in_progress",
                "requires_action",
            ]:
                with st.spinner("필요한 도구를 실행 중입니다..."):
                    if (
                        assistant_client.get_run(run_id, thread_id).status
                        == "requires_action"
                    ):
                        assistant_client.submit_tool_outputs(run_id, thread_id)
                        time.sleep(0.5)
                    else:
                        time.sleep(0.5)

        # 답변 생성 완료 후 새로운 AI 메시지를 표시
        message = (
            assistant_client.get_messages(thread_id)[-1]
            .content[0]
            .text.value.replace("$", "\$")
        )

        # 새로운 답변을 'ai' 메시지로 표시
        with response_placeholder.container():
            st.chat_message("ai").markdown(message)

        # 세션에 메시지 추가
        discussion_client.save_message(message, "ai")
