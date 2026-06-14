import os
import time
import uuid

import requests
import streamlit as st
import logfire
from dotenv import load_dotenv

load_dotenv()


try:
    logfire_token = os.getenv("LOGFIRE_TOKEN", "")

    if not logfire_token:
        print("Logfire token is empty or none!")

    logfire.configure(token=logfire_token, service_name="Studious")
    logfire_status = "Logfire connected"

except Exception as e:
    print(f"Logfire error in Init: {str(e)}")
    logfire_status = f"Logfire error: {str(e)}"

st.set_page_config(
    page_title="Studious",
    layout="wide",
    initial_sidebar_state="expanded",
)

ai_avatar = "🤖"
user_avatar = "👤"

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logfire.info("Session ID created: " + st.session_state.session_id)

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    logfire.info("User ID created: " + st.session_state.user_id)


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.success(logfire_status)
    st.info(f"Memory ID: {st.session_state.session_id[:6]}")

    if st.button("Clear", width="stretch", type="primary"):
        logfire.warn(f"Deleting session: {st.session_state.session_id[:6]}")
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

st.title("Studious - Agentic Assistant")

for message in st.session_state.messages:
    avatar = ai_avatar if message["role"] == "assistant" else user_avatar
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything!"):
    with logfire.span(
        "User chat interaction",
        user_query=prompt,
        session_id=st.session_state.session_id,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=ai_avatar):
            with st.status("Agent is thinking...", expanded=True) as status:
                try:
                    with logfire.span("Calling backend"):
                        base_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                        query_url = f"{base_url}/query"
                        payload = {
                            "question": prompt,
                            "session_id": st.session_state.session_id,
                            "user_id": st.session_state.user_id,
                        }
                        response = requests.post(
                            url=query_url, json=payload, timeout=60
                        )
                        data = response.json()

                    status.update(
                        label="Answer synthesized", state="complete", expanded=False
                    )
                    sources = data.get("sources", [])

                    if sources:
                        with st.expander("View Retrieve Context Sources"):
                            for i, source in enumerate(sources):
                                preview = source[:100].replace("\n", " ") + "..."

                                with st.expander(f"{i+1} -> {preview}"):
                                    st.info(source)

                except Exception as e:
                    logfire.error(f"Connection failed: {str(e)}")
                    status.update(label="Connection failed", state="error")
                    st.error("Offline!!!")
                    st.stop()

            answer_placeholder = st.empty()
            answer = data.get("answer", "No response")

            curr_txt = ""
            for char in answer:
                curr_txt += char
                answer_placeholder.markdown(curr_txt + "|")
                time.sleep(0.005)

            answer_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            logfire.info("Chat completed!")


"""
"answer": state.final_answer,
"sources": state.source_used,
"retrieval_rounds": state.current_round,
"chunks_used": len(state.accepted_chunks),
"sub_questions": state.sub_questions,
"""
