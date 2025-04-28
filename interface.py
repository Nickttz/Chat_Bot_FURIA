import streamlit as st

def iniciar_chat(prompt, llm, retriever):
    st.set_page_config(page_title="Chat FURIA 🐆", page_icon="🔥")

    # Estilo customizado
    st.markdown("""
        <style>
        body { background-color: #0f0f0f; color: #ffffff; }
        .stChatMessage { font-size: 16px; }
        .stMarkdown p { color: #ffffff; }
        .block-container { padding-top: 2rem; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔥 Chat da FURIA Esports")
    st.caption("Converse com a **FURIOSA**, a IA da matilha 🐆")

    if "mensagens" not in st.session_state:
        st.session_state.mensagens = [
            {"role": "assistant", "content": "E aí, torcedor! Eu sou a **FURIOSA**, a inteligência artificial da matilha. Me pergunta algo sobre a FURIA!"}
        ]

    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    pergunta = st.chat_input("Manda sua pergunta sobre a FURIA...")

    if pergunta:
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        with st.chat_message("user"):
            st.markdown(pergunta)

        result_docs = retriever.invoke(pergunta)
        contexto = "\n".join(doc.page_content for doc in result_docs)

        final_prompt = prompt.format(contexto=contexto, pergunta=pergunta)

        resposta = llm.invoke(final_prompt)

        st.session_state.mensagens.append({"role": "assistant", "content": resposta.content})
        with st.chat_message("assistant"):
            st.markdown(resposta.content)
