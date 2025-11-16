# Importando bibliotecas necessárias
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import model_questions

def load_vectorstore():
    persistence_directory = "./chroma_db"
    vectorstore = Chroma(
        persist_directory=persistence_directory,
        embedding_function=OllamaEmbeddings(model="llama3"),
    )
    return vectorstore

def list_docs_names(type_file="all"): # type_file pode ser "all", "docs", "codes", etc.
    vectorstore = load_vectorstore()
    resultados = vectorstore._collection.get(include=["metadatas"])
    arquivos = set()  # usamos set para evitar repetições de chunks do mesmo arquivo

    for metadata in resultados["metadatas"]:
        # O DirectoryLoader geralmente salva o caminho do arquivo em 'source'
        if "source" in metadata:
            # Remove tudo antes de uma barra invertida (\) e depois de um ponto (.)
            arquivo_formatado = metadata["source"].split("/")[-1].split(".")[0]
            if not arquivo_formatado.startswith(type_file) and type_file != "all":
                continue

            if type_file != "all":               
                arquivo_formatado = arquivo_formatado.removeprefix(f"{type_file}-")
            else:
                if(arquivo_formatado.startswith("docs-")):
                    arquivo_formatado = arquivo_formatado.removeprefix("docs-")
                elif(arquivo_formatado.startswith("codes-")):
                    arquivo_formatado = arquivo_formatado.removeprefix("codes-")

            arquivos.add(arquivo_formatado)

    return ", ".join(arquivos)

def define_files_need(question_files, prompt_input, files_in_rag, max_files=5):

    files_supose = model_questions.execute_question(
        question_files,
        prompt_input,
    )

    list_files = [file.strip().lower() for file in files_supose.split(";")]

    # Filtrar os arquivos que estão em files_in_rag
    filtered_files = [file for file in list_files if file in files_in_rag][:max_files]

    # Retornar os arquivos filtrados
    return filtered_files, files_supose

def create_retriever(vectorstore, files_needed, steps: list, k=10):
    if 'filter' in steps and files_needed and len(files_needed) > 0:
        filter_dict = {"source": {"$in": files_needed}}
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, 'filter': filter_dict},
        )
    else:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10},
        )

    return retriever


def main():
    print(list_docs_names("all"))

if __name__ == "__main__":
    main()
