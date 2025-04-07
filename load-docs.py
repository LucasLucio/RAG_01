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

#Baixando pacotes específicos do NLTK (Natural Language Toolkit)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def simple_rag(directory, question):
    # Inicializa o modelo de chat Ollama com o modelo "llama3"
    model = ChatOllama(model="llama3", )
    # Define o diretório de persistência para o banco de dados Chroma
    persistance_directory = "./chroma_db"

    if directory is not None:
        # Carrega documentos do diretório especificado
        docLoad = DirectoryLoader(directory, glob="**/*.pdf", show_progress=True).load()
        # Divide os documentos em pedaços menores
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        splits = text_splitter.split_documents(docLoad)

        if os.path.exists(persistance_directory) and os.path.isdir(persistance_directory):
            # Carrega o banco de dados Chroma existente
            vectorstore = Chroma(persist_directory=persistance_directory, embedding_function=OllamaEmbeddings(model="llama3"))
        else:
            # Cria um novo banco de dados Chroma a partir dos documentos divididos
            vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="llama3"), persist_directory=persistance_directory)
    else:
        if os.path.exists(persistance_directory) and os.path.isdir(persistance_directory):
            # Carrega o banco de dados Chroma existente
            vectorstore = Chroma(persist_directory=persistance_directory, embedding_function=OllamaEmbeddings(model="llama3"))
        else:
            # Imprime uma mensagem de erro se nenhum diretório for fornecido e o banco de dados persistente não existir
            print("Nenhum diretório fornecido e a base de dados persistente não existe.")
            return

    # Configura o recuperador de documentos com base na similaridade
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Define o prompt do sistema para o modelo de chat
    system_prompt = (
        "Considere que você é um assistente de programação com foco em codificação para Uniface."
        "Use os seguintes pedaços de contexto recuperado para responder as solicitações realizadas."
        "Se não souber as respostas ou não conseguir gerar algum código para a solicitação diga que não é possível realizar a operação desejada."
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

    # Imprime a resposta obtida
    print("================== RESPONSE ==================")
    print(response["answer"])
    print("================ END RESPONSE ================")

def main():
    # Solicita ao usuário se deseja carregar os documentos de análise
    loadDocs = input("Deseja carregar os documentos de análise? (s/n): \n")

    # Verifica se o usuário deseja carregar os documentos
    if(loadDocs == 's'):
        # Solicita ao usuário o caminho para os documentos de análise
        directory = input("Caminho para os docs de análise: \n")
    else:
        # Define o diretório como None se o usuário não quiser carregar os documentos
        directory = None
    
    # Solicita ao usuário que digite sua pergunta
    question = input("Digite sua pergunta: \n")

    # Chama a função simple_rag passando o diretório e a pergunta como argumentos
    simple_rag(directory, question)


# Verifica se o script está sendo executado diretamente (não importado como módulo)
if __name__ == "__main__":
    # Chama a função main para iniciar o programa
    main()