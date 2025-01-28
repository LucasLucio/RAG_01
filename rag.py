#Importanto bibliotecas necessárias
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
import os

from langchain_core.prompts import ChatPromptTemplate
import  nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def simple_rag(directory, question):

    model = ChatOllama(model="llama3")
  
    docLoad = DirectoryLoader(directory, glob="**/*.pdf", show_progress=True).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docLoad)
    
    persistance_directory = "./chroma_db"

    if(os.path.exists(persistance_directory) and os.path.isdir(persistance_directory)):
        vectorstore = Chroma(persist_directory = persistance_directory, embedding_function=OllamaEmbeddings(model="llama3"))
    else:
        vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="llama3"), persist_directory = persistance_directory)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    system_prompt = (
        "Considere que você é um assistente de programação com foco em codificação para Uniface."
        "Use os seguintes pedaços de contexto recuperado para responder as solicitações realizadas."
        "Se não souber as respostas ou não conseguir gerar algum código para a solicitação diga que não é possível realizar a operação desejada."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": question})
    response["answer"]

    print("================== RESPONSE ==================")
    print(response["answer"])
    print("================ END RESPONSE ================")


def main():
    directory = input("Caminho para os docs de análise: \n")
    question = input("Digite sua pergunta: \n")
    simple_rag(directory, question)


if __name__ == "__main__":
    main()