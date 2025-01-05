#Importanto bibliotecas necessárias
import streamlit as st
import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

import os 
os.environ['USER_AGENT'] = 'myagent'

#Ciando página para input de fontes
st.title("RAG com página web")
st.caption("Extraia informações de uma página da web usando RAG com Llama-3")

webpage_url = st.text_input("Insira a página web", type="default")


if webpage_url:
    loader = WebBaseLoader(webpage_url)

    #Carrega documentos
    docs = loader.load()

    #Define tamanho de pedaços de texto
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)

    #Divide documentos em pedaços
    splits = text_splitter.split_documents(docs)

    #Cria embrdding
    embeddings = OllamaEmbeddings(model="llama3")

    #Cria vetorstore com documentos quebrados e embeddings
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)


    #Função para perguntar ao modelo
    def ollama_llm(question, context):

        #Froamta pergunta e contexto
        formatted_prompt = f"Question: {question}\n\nContext: {context}"

        #Criando pergunta ao modelo
        response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])

        #Obtendo resposta
        return response['message']['content']

    retriever = vectorstore.as_retriever()

    #Função para adicionar contexto
    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    #Função para chamar RAG
    def rag_chain(question):

        #Recupera documentos
        retrieved_docs = retriever.invoke(question)

        #Formata contexto
        formatted_context = combine_docs(retrieved_docs)
        print(formatted_context)

        #Retorna resposta
        return ollama_llm(question, formatted_context)

    st.success(f"Carregado {webpage_url} com sucesso!")
   
prompt = st.text_input("Pergunte algo para o modelo", type="default")

#Se houver pergunta, chama função do RAG
if prompt:
    #Chama função do RAG
    result = rag_chain(prompt)

    #Mostra resultado
    st.write(result)
