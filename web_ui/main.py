import os
import time
import uuid
from pathlib import Path

import logfire
import requests
import streamlit as st
from dotenv import load_dotenv

from src.utils.constants import ParseMethod

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

    st.divider(width="stretch")
    st.subheader("Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "md", "docx"],
        help="Upload pdf, txt, md, or docx files",
    )

    if uploaded_file is not None:
        parse_method = st.selectbox(
            "Parse Method",
            options=list(ParseMethod),
            format_func=lambda x: x.value.replace("_", " ").title(),
            index=0,
        )

        if st.button("Upload Document", type="primary", use_container_width=True):
            with st.spinner(f"Uploading {uploaded_file.name}..."):
                try:
                    temp_dir = Path("data")
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / uploaded_file.name
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    parse_url = os.getenv("BACKEND_URL", "http://localhost:8000")
                    ingest_url = f"{parse_url}/ingestion"
                    doc_id = str(uuid.uuid4())
                    payload = {
                        "file_path": str(temp_path),
                        "parse_method": parse_method,
                        "doc_id": doc_id,
                    }

                    response = requests.post(url=ingest_url, json=payload, timeout=120)
                    if response.status_code == 200:
                        st.success(f"Successfully uploaded: {uploaded_file.name}")
                        logfire.info(f"Document uploaded: {uploaded_file.name}")
                    else:
                        st.error(f"Upload failed: {response.text}")

                except Exception as e:
                    st.error(f"Error uploading file: {str(e)}")
                    logfire.error(f"Upload error: {str(e)}")

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
                        response = requests.post(url=query_url, json=payload, timeout=60)
                        data = response.json()

                    status.update(label="Answer synthesized", state="complete", expanded=False)
                    sources = data.get("sources", [])

                    if sources:
                        with st.expander("View Retrieve Context Sources"):
                            for i, source in enumerate(sources):
                                preview = source[:100].replace("\n", " ") + "..."

                                with st.expander(f"{i + 1} -> {preview}"):
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
