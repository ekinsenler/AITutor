import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

instruction = """
You are a knowledgeable and supportive tutor chatbot that interacts with students. You are specialized in the Turkish language.
You will receive questions in Turkish, and as a tutor, you will break down the topic step-by-step, providing thorough and clear explanations to help students understand.
You will answer questions accurately by dividing your response into meaningful parts with triple dashes (---) separating each part:
After explaining the topic, you will always ask a follow-up question to validate the student's understanding and encourage them to engage further. If the student asks a follow-up question, you will provide a detailed response to help them grasp the concept.
Remember to be patient and supportive, maintaining a warm and friendly tone throughout the conversation.
If student understands the topic, you are going to test their knowledge by asking a question related to the topic.
Maintain a warm and friendly tone, typical of a helpful tutor.
History: {history}
User: {input}
Assistant:
"""

prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template=instruction
)

def display_parts(response):
    parts = response.split("---")
    for part in parts:
        if part.strip():
            with st.chat_message("assistant"):
                st.markdown(part)
                
def create_chatbot_chain():
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt_template,
        verbose=True
    )
    return chain

def main():
    st.title("Hoşgeldiniz! Öğrenci Yardımcı Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "generated" not in st.session_state:
        st.session_state.generated = []
    if "past" not in st.session_state:
        st.session_state.past = []
    if "chain" not in st.session_state:
        st.session_state.chain = create_chatbot_chain()

    chat_container = st.container()

    with chat_container:
        for i in range(len(st.session_state.past)):
            with st.chat_message("user"):
                st.markdown(st.session_state.past[i])
            with st.chat_message("assistant"):
                st.markdown(st.session_state.generated[i])

    st.write("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    user_input = st.chat_input("Sorunuzu buraya yazınız...")

    if user_input:
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        st.session_state.past.append(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner('Yanıt oluşturuluyor...'):
            response = st.session_state.chain.run(user_input)

        with chat_container:
            display_parts(response)

        st.session_state.generated.append(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    st.write("<div style='height: 50px;'></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()