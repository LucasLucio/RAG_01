#Importando bibliotecas necessárias
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
import os
from langchain_core.prompts import ChatPromptTemplate
import  nltk
import streamlit as st

#Baixando pacotes específicos do NLTK (Natural Language Toolkit)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def simple_rag(question):
    # Inicializa o modelo de chat Ollama com o modelo "llama3"
    model = ChatOllama(model="llama3", )
    # Define o diretório de persistência para o banco de dados Chroma
    persistance_directory = "./chroma_db"

    vectorstore = Chroma(persist_directory=persistance_directory, embedding_function=OllamaEmbeddings(model="llama3"))

    # Configura o recuperador de documentos com base na similaridade
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Define o prompt do sistema para o modelo de chat
    system_prompt = (
        "Considere que você é um assistente de programação com foco em codificação para Uniface."
        "Respondendo a solicitação realizada, sempre codificando em Uniface."
        "Use os seguintes pedaços de contexto recuperado para responder as solicitações realizadas."
        "Se não souber as respostas ou não conseguir gerar algum código para a solicitação diga que não é possível realizar a operação desejada."
        "Responda sempre em Portugues"
        "Idente os blocos de código automaticamente"
        "\n\n"
        "{context}"
    )

    # Cria o template de prompt para o modelo de chat
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # Cria a cadeia de perguntas e respostas usando o modelo e o prompt
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    # Cria a cadeia de recuperação usando o recuperador e a cadeia de perguntas e respostas
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoca a cadeia de recuperação com a pergunta fornecida
    response = rag_chain.invoke({"input": question})
    response["answer"]


#txt_input = input("Digite sua pergunta: \n")

#resposta = simple_rag(txt_input)


# Interface com Streamlit
st.title("IA para codificação em Uniface")

# Entrada de pergunta
txt_input = st.text_input("Como posso te ajudar? :")
if st.button("Consultar"):
    if txt_input:
        with st.spinner('Carregando...'):
            resposta = simple_rag(txt_input)
        st.write(resposta)
    else:
        st.warning("Por favor, digite uma pergunta.")
