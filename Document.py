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


st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

if "api_key_check" not in st.session_state:
    st.session_state["api_key_check"] = False

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "선택해주세요"


API_KEY_pattern = r"sk-.*"

Model_pattern = r"gpt-*"

openai_models = ["선택해주세요", "gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"]

st.title("DocumentGPT")

st.markdown(
    """
    안녕하세요! 이 페이지는 문서를 읽어주는 AI입니다.😄 
    
    문서를 업로드하고 질문을 하면 문서에 대한 답변을 해줍니다.
    """
)


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    os.makedirs("./.cache/files", exist_ok=True)
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/open_ai/{file.name}")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", ".", "?", "!"],
        chunk_size=1000,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=st.session_state["api_key"],
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def save_api_key(api_key):
    st.session_state["api_key"] = api_key
    st.session_state["api_key_check"] = True


def save_openai_model(openai_model):
    st.session_state["openai_model"] = openai_model
    st.session_state["openai_model_check"] = True


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

    api_key = st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
    ).strip()

    if api_key:
        save_api_key(api_key)
        st.write("😄API_KEY가 저장되었습니다.😄")

    button = st.button("저장")

    if button:
        save_api_key(api_key)
        if api_key == "":
            st.warning("OPENAI_API_KEY를 넣어주세요.")

    openai_model = st.selectbox(
        "OpneAI Model을 골라주세요.",
        options=openai_models,
    )
    if openai_model != "선택해주세요":
        if re.match(Model_pattern, openai_model):
            save_openai_model(openai_model)
            st.write("😄모델이 선택되었습니다.😄")

    st.write(
        """
             

        Made by hary.
             
        Github
        https://github.com/lips85/GPT_hary

        streamlit
        https://hary-gpt.streamlit.app/

        """
    )

llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks={
        ChatCallbackHandler(),
    },
    model=st.session_state["openai_model"],
    openai_api_key=st.session_state["api_key"],
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are an AI that reads documents for me. Please answer based on the document given below. 
            If the information is not in the document, answer the question with “The required information is not in the document.” Never make up answers.
            Please answer in the questioner's language 
            
            Context : {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


if (st.session_state["api_key_check"] == True) and (
    st.session_state["api_key"] is not None
):
    if file:
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
                message = "OPENAI_API_KEY or 모델 선택이 잘못되었습니다. 사이드바를 다시 확인하세요."
                send_message(message, "ai")

    else:
        st.session_state["messages"] = []
