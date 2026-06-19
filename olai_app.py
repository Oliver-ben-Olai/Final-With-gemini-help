import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Olai", page_icon="🤖")

st.title("🤖 OLAI")
st.caption("Your Free Cloud AI Assistant")

@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="microsoft/DialoGPT-medium"
    )

bot = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Talk to Olai...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = bot(
                prompt,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )

            reply = response[0]["generated_text"]

            st.write(reply)

            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )
