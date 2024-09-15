# (EN)
# Refactor the agent you made in the previous assignment into an OpenAI Assistant.
# Give it a user interface with Streamlit that displays the conversation history.
# Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar
# Using st.sidebar put a link to the Github repo with the code of your Streamlit app.

# (KR)
# 이전 과제에서 만든 에이전트를 OpenAI 어시스턴트로 리팩터링합니다.
# 대화 기록을 표시하는 Streamlit 을 사용하여 유저 인터페이스를 제공하세요.
# 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 이를 로드합니다.
# st.sidebar를 사용하여 Streamlit app 의 코드과 함께 깃허브 리포지토리에 링크를 넣습니다.

import time
import json
import streamlit as st
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.document_loaders.web_base import WebBaseLoader
from openai import OpenAI
from utils.functions.chat import ChatMemory
from dotenv import load_dotenv
from utils.constant.constant import OPENAI_MODEL
from utils.functions.save_env import SaveEnv
from utils.functions.debug import Debug

load_dotenv()


# 세션 상태 초기화
for key, default in [
    ("messages", []),
    ("api_key", None),
    ("api_key_check", False),
    ("openai_model", "선택해주세요"),
    ("openai_model_check", False),
    ("assistant_id", ""),
    ("assistant", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


st.set_page_config(
    page_title="AssistantGPT",
    page_icon="🚀",
    layout="wide",
)

st.markdown(
    """
    # 🚀 리서치 마스터  🚀 
    
    검색은 저에게 맡겨주세요! 여러분들의 시간을 아껴드리겠습니다.
    (OpenAI Assistant APi 사용)
 """
)


class ThreadClient:
    def __init__(self, client):
        self.client = client

    def get_run(self, run_id, thread_id):
        return self.client.beta.threads.runs.retrieve(
            run_id=run_id,
            thread_id=thread_id,
        )

    def send_message(self, thread_id, content):
        return self.client.beta.threads.messages.create(
            thread_id=thread_id, role="user", content=content
        )

    def get_messages(self, thread_id):
        messages = self.client.beta.threads.messages.list(thread_id=thread_id)
        messages = list(messages)
        messages.reverse()
        for message in messages:
            if message.role == "user":
                discussion_client.send_message(message.content[0].text.value, "user")

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
        discussion_client.send_message("이슈를 찾았어요!", "ai")
        discussion_client.send_message(outputs[0]["output"], "ai")

        return self.client.beta.threads.runs.submit_tool_outputs(
            run_id=run_id,
            thread_id=thread_id,
            tool_outputs=outputs,
        )

    def wait_on_run(self, run, thread):
        while run.status == "queued" or run.status == "in_progress":
            run = self.get_run(run.id, thread.id)
            time.sleep(0.5)
        return run


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


issue_search_client = IssueSearchClient()
discussion_client = ChatMemory()

functions_map = {
    "get_websites_by_wikipedia_search": issue_search_client.get_websites_by_wikipedia_search,
    "get_websites_by_duckduckgo_search": issue_search_client.get_websites_by_duckduckgo_search,
    "get_document_text": issue_search_client.get_document_text,
}

functions = [
    {
        "type": "function",
        "function": {
            "name": "get_websites_by_wikipedia_search",
            "description": "Use this tool to find the websites for the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for. Example query: Research about the XZ backdoor",
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
            "description": "Use this tool to find the websites for the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query you will search for. Example query: Research about the XZ backdoor",
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
            "description": "Use this tool to load the website for the given url.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url you will load. Example url: https://en.wikipedia.org/wiki/Backdoor_(computing)",
                    }
                },
                "required": ["url"],
            },
        },
    },
]

with st.sidebar:
    # API Key 입력 필드
    st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        on_change=SaveEnv.save_api_key,  # 인스턴스 메서드로 변경
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
        on_change=SaveEnv.save_openai_model,  # 인스턴스 메서드로 변경
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
    client = OpenAI(api_key=st.session_state["api_key"])
    assistant = client.beta.assistants.create(
        name="Research Assistant",
        instructions="You help users do research on keyword from wikipedia and duckduckgo.",
        model="gpt-4o-mini-2024-07-18",
        tools=functions,
    )
    assistant_id = assistant.id
    st.session_state["assistant_id"] = assistant_id

    discussion_client.send_message("무엇이든 물어보세요!", "ai", save=False)
    discussion_client.paint_history()
    query = st.chat_input("Ask a question to the website.")

    if query:
        st.session_state["query"] = query
        discussion_client.send_message(query, "human")
        thread = client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": f"{query}",
                }
            ]
        )
        thread_id = thread.id
        run = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
        run_id = run.id

        assistant = ThreadClient(client)
        run = assistant.wait_on_run(run, thread)

        if run:
            discussion_client.send_message("이슈를 찾고 있어요!", "ai", save=False)
            discussion_client.paint_history()
            assistant.get_tool_outputs(run_id, thread_id)
            assistant.submit_tool_outputs(run_id, thread_id)
            st.download_button(
                label="채팅 내역 다운로드",
                data=json.dumps(st.session_state["messages"]),
                file_name="chat_history.txt",
                mime="text/plain",
            )

    else:
        st.session_state["messages"] = []
