import re
import os
import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.storage import LocalFileStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.callbacks.base import BaseCallbackHandler

# Streamlit 페이지 설정
st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
    layout="wide",
)

# 세션 상태 초기화
for key, default in [
    ("messages", []),
    ("api_key", None),
    ("api_key_check", False),
    ("openai_model", "선택해주세요"),
    ("openai_model_check", False),
    ("file", None),
    ("file_check", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# 정규 표현식 패턴
API_KEY_pattern = r"sk-.*"
Model_pattern = r"gpt-*"

# OpenAI 모델 목록
openai_models = ["선택해주세요", "gpt-4o-mini-2024-07-18", "gpt-4o-2024-08-06"]

# 페이지 제목 및 설명
st.title("DocumentGPT")

if not (
    st.session_state["api_key_check"]
    and st.session_state["openai_model_check"]
    and st.session_state["file_check"]
):
    st.markdown(
        """
        안녕하세요! 이 페이지는 문서를 읽어주는 AI입니다.😄 
        
        문서를 업로드하고 질문을 하면 문서에 대한 답변을 해줍니다.
        """
    )
    if not st.session_state["api_key_check"]:
        st.warning("API_KEY를 넣어주세요.")
    if not st.session_state["openai_model_check"]:
        st.warning("모델을 선택해주세요.")
    if not st.session_state["file_check"]:
        st.warning("문서를 업로드해주세요.")
else:
    st.success("😄API_KEY와 모델이 저장되었습니다.😄")


# 콜백 핸들러 클래스
class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# 파일 임베딩 함수
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    os.makedirs("./.cache/files", exist_ok=True)
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())

    cache_dir = LocalFileStore(f"./.cache/embeddings/open_ai/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", ".", "?", "!"],
        chunk_size=1000,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=st.session_state["api_key"])
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


# 파일 업로드 체크 함수
def save_file():
    if st.session_state["file"]:
        st.session_state["file_check"] = True
    else:
        st.session_state["file_check"] = False


# 메시지 저장 함수
def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


# 메시지 전송 함수
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


# 채팅 기록 표시 함수
def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


# API 키 저장 함수
def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_check"] = True


# OpenAI 모델 저장 함수
def save_openai_model(openai_model):
    st.session_state["openai_model"] = openai_model
    st.session_state["openai_model_check"] = True


# 사이드바 설정
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
        on_change=save_file,
        key="file",
    )
    if st.session_state["file_check"]:
        st.success("😄문서가 업로드되었습니다.😄")
    else:
        st.warning("문서를 업로드해주세요.")

    api_key = st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        on_change=lambda: (
            save_api_key(st.session_state["api_key"])
            if re.match(API_KEY_pattern, st.session_state["api_key"])
            else None
        ),
        key="api_key",
    )

    if st.session_state["api_key_check"]:
        st.success("😄API_KEY가 저장되었습니다.😄")
    else:
        st.warning("API_KEY를 넣어주세요.")

    openai_model = st.selectbox(
        "OpenAI Model을 골라주세요.",
        options=openai_models,
        on_change=lambda: save_openai_model(st.session_state["openai_model"]),
        key="openai_model",
    )

    if st.session_state["openai_model_check"]:
        st.success("😄모델이 선택되었습니다.😄")
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

# 메인 로직
if (
    st.session_state["api_key_check"]
    and st.session_state["file_check"]
    and st.session_state["openai_model_check"]
):

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
        model=st.session_state["openai_model"],
        openai_api_key=st.session_state["api_key"],
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an AI that reads documents for me. Please answer based on the document given below. 
                If the information is not in the document, answer the question with "The required information is not in the document." Never make up answers.
                Please answer in the questioner's language 
                
                Context : {context}
                """,
            ),
            ("human", "{question}"),
        ]
    )

    if st.session_state["file_check"]:
        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")

        if message:
            if re.match(API_KEY_pattern, st.session_state["api_key"]) and re.match(
                Model_pattern, st.session_state["openai_model"]
            ):
                send_message(message, "human")
                chain = (
                    {
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                )
                try:
                    with st.chat_message("ai"):
                        chain.invoke(message)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.warning("OPENAI_API_KEY or 모델 선택을 다시 진행해주세요.")
            else:
                send_message(
                    "OPENAI_API_KEY or 모델 선택이 잘못되었습니다. 사이드바를 다시 확인하세요.",
                    "ai",
                )
    else:
        st.session_state["messages"] = []
