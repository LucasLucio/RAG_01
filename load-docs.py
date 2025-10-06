# Importando bibliotecas necessárias
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document


def load_base(directory):
    # Define o diretório de persistência para o banco de dados Chroma
    persistence_directory = "./chroma_db"

    if directory is not None:
        # Carrega documentos do diretório especificado
        print(f"Carregando documentos do diretório: {directory}")
        doc_loader = DirectoryLoader(directory, glob="**/*.pdf", show_progress=True)
        documents = doc_loader.load()

        # Divide os documentos em pedaços menores (se necessário)
        print("Dividindo documentos em chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=600)
        splits = text_splitter.split_documents(documents)

        # Cria (ou atualiza) o banco de dados Chroma
        print("Criando ou atualizando a base vetorial...")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=OllamaEmbeddings(model="llama3"),
            persist_directory=persistence_directory
        )

        num_docs = vectorstore._collection.count()  # número de vetores/documentos
        print(f"\n Carregamento concluído! Total de documentos vetorizados: {num_docs}")


    else:
        print("Nenhum diretório fornecido. Informe o caminho contendo os PDFs.")


def main():
    directory = input("Caminho para os docs de análise: \n")
    load_base(directory)


if __name__ == "__main__":
    main()