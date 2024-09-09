import json
import streamlit as st
import re
import os

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.retrievers import WikipediaRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 파일 분리 (상수들)
from utils.constant.constant import OPENAI_MODEL, API_KEY_PATTERN


st.set_page_config(
    page_title="QuizGPT❓❗️",
    page_icon="❓",
)
# 세션 상태 초기화
for key, default in [
    ("messages", []),
    ("api_key", None),
    ("api_key_check", False),
    ("openai_model", "선택해주세요"),
    ("openai_model_check", False),
    ("quiz_subject", ""),
    ("quiz_submitted", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# API 키 저장 함수
def save_api_key():
    if re.match(API_KEY_PATTERN, st.session_state["api_key"]):
        st.session_state["api_key_check"] = True


# OpenAI 모델 저장 함수
def save_openai_model():
    st.session_state["openai_model_check"] = (
        st.session_state["openai_model"] != "선택해주세요"
    )


def set_quiz_submitted(value: bool):
    st.session_state.update({"quiz_submitted": value})


def embed_file(file):
    os.makedirs("./.cache/files", exist_ok=True)
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        separators=["\n\n", ".", "?", "!"],
        chunk_size=1000,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_data(show_spinner="퀴즈 나옵니다...")
def run_quiz_chain(_docs, subject, count, difficulty):
    chain = prompt | llm
    return chain.invoke(
        {
            "docs": _docs,
            "subject": subject,
            "count": count,
            "difficulty": difficulty,
        }
    )


def reset_quiz():
    st.session_state["quiz_subject"] = ""
    st.session_state["quiz_submitted"] = False
    run_quiz_chain.clear()


# 페이지 제목 및 설명
st.title("QuizGPT❓❗️")

if not (st.session_state["api_key_check"] and st.session_state["openai_model_check"]):
    st.markdown(
        """
    안녕하세요! 이 페이지는 문서를 읽어주는 AI입니다.😄 
    
    문서를 업로드하고 질문을 하면 문서에 대한 답변을 해줍니다.
    """
    )
else:
    st.success("😄API_KEY와 모델이 저장되었습니다.😄")


with st.sidebar:
    docs = None
    topic = None

    choice = st.selectbox(
        "파일 또는 위키피디아 검색을 선택해주세요.",
        (
            "선택하세요",
            "파일",
            "위키피디아",
        ),
    )
    if choice == "파일":
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
            key="file",
        )
        if file:
            docs = embed_file(file)

    st.text_input(
        "API_KEY 입력",
        placeholder="sk-...",
        on_change=save_api_key,
        key="api_key",
    )

    if st.session_state["api_key_check"]:
        st.success("😄API_KEY가 저장되었습니다.😄")
    else:
        st.warning("API_KEY를 넣어주세요.")

    st.selectbox(
        "OpenAI Model을 골라주세요.",
        options=OPENAI_MODEL,
        on_change=save_openai_model,
        key="openai_model",
    )

    if st.session_state["openai_model_check"]:
        st.success("😄모델이 선택되었습니다.😄")
    else:
        st.warning("모델을 선택해주세요.")

    st.divider()
    st.markdown(
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
    # and st.session_state["quiz_submitted"]
):
    if not st.session_state["api_key_check"]:
        st.warning(":blue[OpenAI API Key]를 넣어주세요.")
    else:
        st.success("😄API_KEY가 저장되었습니다.😄")
    if not st.session_state["openai_model_check"]:
        st.warning(":blue[OpenAI 모델]을 선택해주세요.")
    else:
        st.success("😄:blue[OpenAI 모델]이 선택되었습니다.😄")

else:

    function = {
        "name": "create_quiz",
        "description": "function that takes a list of questions and answers and returns a quiz",
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                            },
                            "answers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "answer": {
                                            "type": "string",
                                        },
                                        "correct": {
                                            "type": "boolean",
                                        },
                                    },
                                    "required": ["answer", "correct"],
                                },
                            },
                        },
                        "required": ["question", "answers"],
                    },
                }
            },
            "required": ["questions"],
        },
    }

    llm = ChatOpenAI(
        temperature=0.1,
        streaming=True,
        model=st.session_state["openai_model"],
        openai_api_key=st.session_state["api_key"],
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )

    prompt = PromptTemplate.from_template(
        """            
        Please create a quiz based on the following criteria:

        Document: {docs}
        Topic: {subject}
        Number of Questions: {count}
        Difficulty Level: Level-{difficulty}/5
        Language: Korean

        The quiz should be well-structured with clear questions and correct answers.
        Ensure that the questions are relevant to the specified topic and adhere to the selected difficulty level.
        The quiz format should be multiple-choice,
        and each question should be accompanied by four possible answers, with only one correct option.
        """,
    )

    try:
        col1, col2 = st.columns([5, 1])

        with col1:
            st.markdown(
                """
                #### 자~ 이제 퀴즈를 만들어 볼까요?
                """
            )
        with col2:
            st.button(":red[퀴즈 초기화]", on_click=reset_quiz)

        with st.form("quiz_create_form"):
            col1, col2, col3 = st.columns([5, 1, 1])

            with col1:
                quiz_subject = st.text_input(
                    ":blue[주제]",
                    placeholder="무엇을 주제로 퀴즈를 만들까요?",
                    key="quiz_subject",
                )

            with col2:
                quiz_count = st.number_input(
                    ":blue[개수]",
                    value=10,
                    min_value=2,
                    key="quiz_count",
                )

            with col3:
                difficulty_options = ["1", "2", "3", "4", "5"]
                quiz_difficulty = st.selectbox(
                    ":blue[레벨]",
                    options=difficulty_options,
                    index=0,
                    key="quiz_difficulty",
                )

            submitted = st.form_submit_button(
                "**:blue[퀴즈 만들기 시작]**",
                use_container_width=True,
                on_click=set_quiz_submitted,
                args=(False,),
            )

        if quiz_subject != "":
            response_box = st.empty()
            response = run_quiz_chain(
                _docs=docs if docs else wiki_search(quiz_subject),
                subject=quiz_subject,
                count=quiz_count,
                difficulty=quiz_difficulty,
            )
            response = response.additional_kwargs["function_call"]["arguments"]
            response = json.loads(response)

            generated_quiz_count = len(response["questions"])

            with st.form("quiz_questions_form"):
                solved_count = 0
                correct_count = 0
                answer_feedback_box = []
                answer_feedback_content = []

                for index, question in enumerate(response["questions"]):
                    st.write(f"{index+1}. {question['question']}")
                    value = st.radio(
                        "Select an option.",
                        [answer["answer"] for answer in question["answers"]],
                        index=None,
                        label_visibility="collapsed",
                        key=f"[{quiz_subject}_{quiz_count}_{quiz_difficulty}]question_{index}",
                    )

                    answer_feedback = st.empty()
                    answer_feedback_box.append(answer_feedback)

                    if value:
                        solved_count += 1

                        if {"answer": value, "correct": True} in question["answers"]:
                            answer_feedback_content.append(
                                {
                                    "index": index,
                                    "correct": True,
                                    "feedback": "정답! :100:",
                                }
                            )

                            correct_count += 1
                        else:
                            answer_feedback_content.append(
                                {
                                    "index": index,
                                    "correct": False,
                                    "feedback": "다시 도전해 보아요! :sparkles:",
                                }
                            )

                is_quiz_all_submitted = solved_count == generated_quiz_count

                if is_quiz_all_submitted:
                    for answer_feedback in answer_feedback_content:
                        index = answer_feedback["index"]
                        with answer_feedback_box[index]:
                            if answer_feedback["correct"]:
                                st.success(answer_feedback["feedback"])
                            else:
                                st.error(answer_feedback["feedback"])

                st.divider()

                result = st.empty()

                st.form_submit_button(
                    (
                        "**:blue[제출하기]**"
                        if solved_count < generated_quiz_count
                        else (
                            "**:blue[:100: 축하합니다~ 새로운 주제로 도전해 보세요!]**"
                            if correct_count == generated_quiz_count
                            else "**:blue[다시 도전 💪]**"
                        )
                    ),
                    use_container_width=True,
                    disabled=correct_count == generated_quiz_count,
                    on_click=set_quiz_submitted,
                    args=(True,),
                )

                if st.session_state["quiz_submitted"]:

                    if not is_quiz_all_submitted:
                        result.error(
                            f"퀴즈를 모두 풀고 제출해 주세요. ( 남은 퀴즈 개수: :red[{generated_quiz_count - solved_count}] / 답변한 퀴즈 개수: :blue[{solved_count}] )"
                        )
                    else:
                        result.subheader(
                            f"결과: :blue[{correct_count}] / {generated_quiz_count}"
                        )

                    if correct_count == generated_quiz_count:
                        for _ in range(3):
                            st.balloons()

            if correct_count == generated_quiz_count:
                st.button(
                    ":red[새로운 주제로 도전하기]",
                    use_container_width=True,
                    on_click=reset_quiz,
                )

    except Exception as e:
        st.error(f"An error occurred: {e}")

        if "response" in locals():
            response_box.json(response)
