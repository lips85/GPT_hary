import re
import os
import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.memory.buffer import ConversationBufferMemory

# 파일 분리 (상수들)
from utils.constant.constant import OPENAI_MODEL, API_KEY_PATTERN, MODEL_PATTERN

# 파일 분리 (함수들)
from utils.functions.chat import ChatMemory, ChatCallbackHandler
from utils.functions.save_env import SaveEnv

# 디버그용
from dotenv import load_dotenv

load_dotenv()


# 세션 상태 초기화
for key, default in [
    ("messages", []),
    ("api_key", None),
    ("api_key_check", False),
    ("openai_model", "선택해주세요"),
    ("openai_model_check", False),
    ("url", None),
    ("url_check", False),
    ("url_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
    layout="wide",
)

# 페이지 제목 및 설명
st.title("SiteGPT 🖥️")

if not (
    st.session_state["api_key_check"]
    and st.session_state["openai_model_check"]
    and st.session_state["url_check"]
):
    st.markdown(
        """
        # SiteGPT

        사이트에 대해 질문해 보세요!!

        왼쪽 사이드바에 사이트를 추가해 보세요!!
    """
    )


# 문서 포맷�� 함수
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# 디버깅용 지우는 함수
class Debug:
    @staticmethod
    def my_api_key():
        st.session_state["api_key"] = os.environ["OPENAI_API_KEY"]
        st.session_state["api_key_check"] = True

    @staticmethod
    def my_url():
        st.session_state["url"] = os.environ.get(
            "CLAUDEFLARE_SITEMAP_URL", "https://developers.cloudflare.com/sitemap-0.xml"
        )
        st.session_state["url_check"] = True


# 웹사이트 로딩 및 벡터 저장소 생성 함수
@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    os.makedirs("./.cache/sitemap", exist_ok=True)
    cache_dir = LocalFileStore(
        f"./.cache/sitemap/embeddings/{st.session_state['url_name']}"
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        filter_urls=[
            r"https:\/\/developers.cloudflare.com/ai-gateway.*",
            r"https:\/\/developers.cloudflare.com/vectorize.*",
            r"https:\/\/developers.cloudflare.com/workers-ai.*",
        ],
    )
    loader.requests_per_second = 50
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.session_state["api_key"],
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    history = inputs["history"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "history": history,
    }


# 최적의 답변 선택 함수
def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    history = inputs["history"]
    choose_chain = choose_prompt | llm_for_last
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
            "history": history,
        }
    )


# HTML 페이지 파싱 함수
def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


# 메모리에서 대화 기록 로드 함수
def load_memory(_):
    return memory.load_memory_variables({})["history"]


# 답변 생성을 위한 프롬프트 템플릿
answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.

    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}

    Examples:

    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5

    Question: How far away is the sun?
    Answer: I don't know
    Score: 0

    Your turn!

    Question: {question}

"""
)


# 최종 답변 선택을 위한 프롬프트 템플릿
choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


# 최종 답변 생성을 위한 LLM 모델 설정
llm_for_last = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks={
        ChatCallbackHandler(),
    },
    model=st.session_state["openai_model"],
    openai_api_key=st.session_state["api_key"],
)

# 답변 생성을 위한 LLM 모델 설정
llm = ChatOpenAI(
    temperature=0.1,
    model=st.session_state["openai_model"],
    openai_api_key=st.session_state["api_key"],
)

# 대화 기록을 저장하기 위한 메모리 설정
memory = ConversationBufferMemory(
    llm=llm,
    max_token_limit=1000,
    return_messages=True,
    memory_key="history",
)


with st.sidebar:
    st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        on_change=SaveEnv.save_api_key,
        key="api_key",
        type="password",
    )

    if st.session_state["api_key_check"]:
        st.success("😄API_KEY가 저장되었습니다.😄")
    else:
        st.warning("API_KEY를 넣어주세요.")

    st.button(
        "hary의 API_KEY (디버그용)",
        on_click=Debug.my_api_key,
        key="my_key_button",
    )

    st.divider()

    st.selectbox(
        "OpenAI Model을 골라주세요.",
        options=OPENAI_MODEL,
        on_change=SaveEnv.save_openai_model,
        key="openai_model",
    )

    if st.session_state["openai_model_check"]:
        st.success("😄모델이 선택되었습니다.😄")
    else:
        st.warning("모델을 선택해주세요.")

    st.divider()

    st.text_input(
        "Write down a URL",
        placeholder="https://example.com/sitemap.xml",
        key="url",
        on_change=SaveEnv.save_url,
    )

    if st.session_state["url_check"]:
        st.success("😄URL이 저장되었습니다.😄")
    else:
        st.warning("URL을 넣어주세요.")

    st.button(
        "디버그용 url",
        on_click=Debug.my_url,
        key="my_url_button",
    )

    st.write(
        """
        Made by hary.
        
        Github
        https://github.com/lips85/GPT_hary

        streamlit
        https://hary-gpt.streamlit.app/
        """
    )


if not (
    st.session_state["api_key_check"]
    and st.session_state["openai_model_check"]
    and st.session_state["url_check"]
):

    if not st.session_state["api_key_check"]:
        st.warning("Please provide an **:blue[OpenAI API Key]** on the sidebar.")

    if not st.session_state["openai_model_check"]:
        st.warning("Please write down a **:blue[OpenAI Model Select]** on the sidebar.")

    if not st.session_state["url_check"]:
        st.warning("Please write down a **:blue[Sitemap URL]** on the sidebar.")
else:
    if st.session_state["url_check"]:
        if ".xml" not in st.session_state["url"]:
            with st.sidebar:
                st.error("Please write down a Sitemap URL.")
        else:
            retriever = load_website(st.session_state["url"])
            ChatMemory.send_message("I'm ready! Ask away!", "ai", save=False)
            ChatMemory.paint_history()
            message = st.chat_input("Ask a question to the website.")
            if message:
                if re.match(API_KEY_PATTERN, st.session_state["api_key"]) and re.match(
                    MODEL_PATTERN, st.session_state["openai_model"]
                ):
                    ChatMemory.send_message(message, "human")
                    try:
                        chain = (
                            {
                                "docs": retriever,
                                "question": RunnablePassthrough(),
                                "history": RunnableLambda(load_memory),
                            }
                            | RunnableLambda(get_answers)
                            | RunnableLambda(choose_answer)
                        )

                        def invoke_chain(question):
                            result = chain.invoke(question).content
                            memory.save_context(
                                {"input": question},
                                {"output": result},
                            )
                            return result

                        with st.chat_message("ai"):
                            invoke_chain(message)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

                else:
                    message = "OPENAI_API_KEY or 모델 선택이 잘못되었습니다. 사이드바를 다시 확인하세요."
                    ChatMemory.send_message(message, "ai")
    else:
        st.session_state["messages"] = []
