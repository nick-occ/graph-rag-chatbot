import os
import requests
import streamlit as st

# CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/article-rag-agent")

CHATBOT_URL = "https://graph-rag-chatbot.onrender.com/"

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the 
        [Urban Institute Articles](https://ui.charlotte.edu/articles-research/).
        The agent uses  retrieval-augment generation (RAG).
        """
    )

st.title("Urban Institute Articles Chatbot")
st.info("Ask me questions about the Urban Institute?")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        try:
            response = requests.post(CHATBOT_URL, json=data)

            result = response.json()

            output_text = result.get("output", "No output returned.")
            explanation = result.get("intermediate_steps", [])
            context = result.get("context", [])

        except requests.exceptions.RequestException as e:
            output_text = "Could not connect to the chatbot server. Please make sure it is running."
            explanation = str(e)

    st.chat_message("assistant").markdown(output_text)

    articles = list(
        {(doc["metadata"]["title"], doc["metadata"]["url"]) for doc in context}
    )

    with st.status("Articles", state="complete"):
        for title, url in articles:
            st.markdown(f"- [{title}]({url})")

    with st.status("How was this generated", state="complete"):
        for step in explanation:
            st.markdown(f"- {step}")

    st.session_state.messages.append(
        {"role": "assistant", "output": output_text, "explanation": explanation}
    )
