# Importando bibliotecas necessárias
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

# Função RAG com ajuste automático de top_k
def rag(question):

    model = ChatOllama(model="llama3", temperature=0.5)
    persistence_directory = "./chroma_db"
    vectorstore = Chroma(
        persist_directory=persistence_directory,
        embedding_function=OllamaEmbeddings(model="llama3")
    )

    # Recupera inicialmente k_max documentos
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10, "fetch_k": 30})
 

    # Cria prompt de sistema
    system_prompt = (
        "Você é um assistente de programação especializado em codifcação Uniface.\n"
        "Use os exemplos de código recuperados para gerar respostas corretas.\n"
        "Se não for possível realizar a operação, diga que não é possível.\n"
        "Sempre responda em português e formate os blocos de código corretamente.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    # Cria a cadeia de documentos + modelo
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoca a cadeia passando apenas os documentos selecionados
    response = rag_chain.invoke({"input": question})
    print(response)
    return response["answer"]


# ===== Streamlit =====
st.title("Assistente de Programação Uniface")

txt_input = st.text_input("Como posso te ajudar? :")

if st.button("Consultar"):
    if txt_input:
        with st.spinner('Consultando a base e gerando código...'):
            resposta = rag(txt_input)
        st.code(resposta, language="uniface")
    else:
        st.warning("Por favor, digite uma pergunta.")
