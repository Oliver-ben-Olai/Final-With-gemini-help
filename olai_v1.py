import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="OLAI", page_icon="🤖")

st.title("🤖 OLAI")
st.subheader("Powered by Ollie")
st.caption("Your Free Cloud AI Assistant")

SYSTEM_PROMPT = '''
You are OLAI, a friendly AI assistant.

Your mascot is Ollie, a cute robot.
You are helpful, intelligent, creative, and encouraging.
You answer clearly and conversationally.
You never say you are DialoGPT.
You always introduce yourself as OLAI when asked.
'''

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
        with st.spinner("Ollie is thinking..."):
            conversation = SYSTEM_PROMPT + "\n\n"

            for msg in st.session_state.messages:
                conversation += f"{msg['role']}: {msg['content']}\n"

            response = bot(
                conversation,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8
            )

            reply = response[0]["generated_text"]

            if conversation in reply:
                reply = reply.replace(conversation, "").strip()

            st.write(reply)

            st.session_state.messages.append(
                {"role": "assistant", "content": reply}
            )
