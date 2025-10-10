from classes import ExecutionRag, FilesInRag
# Importando bibliotecas necessárias
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import base_manager

def pre_processing_question(question) -> FilesInRag:

    files_in_rag = FilesInRag()
    # Lista arquivos disponíveis
    rag_files = base_manager.list_docs_names("docs")

    files_in_rag.files_available = [file for file in rag_files.split(",")]


    # Prompt e ajuste de input para selecionar arquivos
    question_files = (
        f"Para o seguinte questionamento do usuário {question},"
        f"Analise dentro destes arquivos disponíveis ({rag_files}) "
        "quais acredita conter explicações da tecnologia Uniface que irão auxiliar "
        "na elaboração de uma resposta completa, concisa e correta."
    )
    prompt_input = (
        "Considere que você é um assistente de perguntas e respostas especializado na tecnologia Uniface"
        "Responda apenas e exatamente o solicitado"
        "Responda apresentando somente uma lista simples com os nomes de arquivos separados por ;"
        "Não crie novos arquivos, utilize somente os apresentados"
        "Não acrescente nada ao início nem ao final da resposta, apenas os nomes dos arquivos."
    )

    # Define os arquivos que serão utilizados
    files_need = base_manager.define_files_need(
        question_files,
        prompt_input,
        rag_files
    )

    files_need = [f"files-input/docs-{file}.pdf" for file in files_need]

    files_in_rag.files_defined = files_need

    return files_in_rag

def rag_docs(question) -> ExecutionRag:

    execution_rag = ExecutionRag()

    model = ChatOllama(model="llama3", temperature=0.6)

    vectorstore = base_manager.load_vectorstore()

    process_files = pre_processing_question(question)
    files_needed = process_files.files_defined

    execution_rag.files_used = process_files

    filter_dict = {"source": {"$in": files_needed}}

    # Recupera inicialmente k_max documentos
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 15, 'filter': filter_dict},
    )

    # Cria prompt de sistema
    system_prompt = (
        "Você é um assistente de perguntas e respostas especializado na tecnologia Uniface.\n"
        "Responda a solicitação do usuário com base nos documentos fornecidos.\n"
        "Utilize a base de conhecimento contida nos documentos para responder a pergunta do usuário.\n"
        "Mantenha o foco na tecnologia Uniface.\n"
        "Baseie suas respostas apenas sobre as informações contidas nos documentos.\n"
        "Caso seja necessário, utilize exemplos de código contidos nos documentos para ilustrar suas respostas.\n"
        "Sempre envie uma resposta que atenda diretamente o que foi solicitado, sem realizar novas perguntas ou solicitações para que o usuário complemente a solicitação original."
        "Se não for possível realizar a resposta para o questionamento passado, diga que não é possível responder por não possuir o conhecimento necessário sobre o assunto para enviar uma resposta assertiva.\n"
        "Sempre responda em português e formate os blocos de código do exemplo corretamente.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # Cria a cadeia de documentos + modelo
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoca a cadeia passando apenas os documentos selecionados
    response = rag_chain.invoke({"input": question})

    execution_rag.question = question
    execution_rag.response = response["answer"]
    execution_rag.context = response["context"]

    return execution_rag