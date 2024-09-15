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

import re
import time
import json
import streamlit as st
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain.utilities.wikipedia import WikipediaAPIWrapper
from langchain.document_loaders.web_base import WebBaseLoader
import openai

# 클라우드페어 공식문서 사이트맵?
# https://developers.cloudflare.com/sitemap.xml


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
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "api_key_check" not in st.session_state:
    st.session_state["api_key_check"] = False

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "선택해주세요"

if "assistant" not in st.session_state:
    st.session_state["assistant"] = ""

API_KEY_pattern = r"sk-.*"

Model_pattern = r"gpt-*"

openai_models = ["선택해주세요", "gpt-3.5-turbo-0125", "gpt-4-0125-preview"]


# class가 안됨 ㅠㅠ 왜인지 모르것음 self를 빼야 돌아감...
# class api_key_and_model:
#     def __init__(self):
#         pass


def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_check"] = True


def save_openai_model(openai_model):
    st.session_state["openai_model"] = openai_model
    st.session_state["openai_model_check"] = True


class DiscussionClient:

    def __init__(self):
        pass

    def save_message(self, message, role):
        st.session_state["messages"].append({"message": message, "role": role})

    def send_message(self, message, role, save=True):
        with st.chat_message(role):
            st.markdown(message)

        if save:
            self.save_message(message, role)

    def paint_history(self):
        for message in st.session_state["messages"]:
            self.send_message(message["message"], message["role"], save=False)


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
discussion_client = DiscussionClient()

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
    api_key = st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        disabled=st.session_state["api_key"] is not None,
    ).strip()

    if api_key:
        save_api_key(api_key=api_key)
        st.write("😄API_KEY가 저장되었습니다.😄")

    button = st.button("저장")

    if button:
        save_api_key(api_key=api_key)
        if api_key == "":
            st.warning("OPENAI_API_KEY를 넣어주세요.")

    st.divider()

    openai_model = st.selectbox(
        "OpneAI Model을 골라주세요.",
        options=openai_models,
    )
    if openai_model != "선택해주세요":
        if re.match(Model_pattern, openai_model):
            save_openai_model(openai_model=openai_model)
            st.write("😄모델이 선택되었습니다.😄")

    client = openai.OpenAI(api_key=st.session_state["api_key"])

    st.write(
        """


        Made by hary.

        Github
        https://github.com/lips85/normard-langchain/blob/main/pages/06_FinalAssitant.py

        streamlit
        https://nomad-fullstack-langchain-hary.streamlit.app/FinalAssitant

        """
    )

if not api_key:
    st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

if openai_model == "선택해주세요":
    st.warning("Please write down a **:blue[OpenAI Model Select]** on the sidebar.")

if api_key and (openai_model != "선택해주세요"):
    st.session_state["api_key"] = api_key
    client = openai.OpenAI(api_key=api_key)

    assistant_id = "asst_kV62UlOmxZsV9WcJE3Npy8t1"

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

        # assistant.get_tool_outputs(run_id, thread_id)
    ##########################################################
    # is_new_result = False

    # with st.chat_message("ai"):
    #     with st.status(":red[Polling Run Status...]") as status:

    #         try:
    #             while True:

    #                 if assistant.get_run(run_id, thread_id).status == "completed":
    #                     is_new_result = True
    #                     status.update(label=f"Running: {assistant.get_run(run_id, thread_id).status}", state="complete")
    #                     break
    #                 if assistant.get_run(run_id, thread_id).status == "requires_action":
    #                     status.update(label=f"Running: {assistant.get_run(run_id, thread_id).status}", state="running")
    #                     assistant.submit_tool_outputs(run_id, thread_id)

    #         except Exception as e:
    #             st.error(f"Error: {e}")

    # if is_new_result:
    #     result = assistant.get_messages(thread_id)
    #     discussion_client.send_message(result, "ai")
    #     st.download_button(
    #         label="채팅 내역 다운로드",
    #         data=json.dumps(st.session_state["messages"]),
    #         file_name="chat_history.txt",
    #         mime="text/plain",
    #     )
    #     st.rerun()

    ####################################
