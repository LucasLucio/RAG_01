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
        """
            Você é um assistente de perguntas e respostas especializado na tecnologia Uniface.
                Responda apenas e exatamente ao que for solicitado.

            Regras:
                - Retorne somente uma lista simples contendo 4 nomes de arquivos, separados por ponto e vírgula (;).
                - Os arquivos devem ser ordenados do mais relevante ao menos relevante, considerando o que foi solicitado.
                - Não invente, não modifique e não crie novos arquivos. Utilize apenas os fornecidos no contexto.
                - Não adicione nenhum texto antes ou depois da lista (sem explicações, comentários, títulos ou formatação extra).
        """
    )

    # Define os arquivos que serão utilizados
    files_need = base_manager.define_files_need(
        question_files,
        prompt_input,
        rag_files
    )

    files_need = [f"files-input/docs/docs-{file}.pdf" for file in files_need]

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
        search_type="similarity",
        search_kwargs={"k": 10, 'filter': filter_dict},
    )

 
    # Cria prompt de sistema
    system_prompt = (
        """
            Você é um assistente especialista na tecnologia Uniface.
            Responda somente com base nas informações presentes nos documentos fornecidos pelo RAG.

            Regras:
            - Foque apenas em assuntos relacionados à tecnologia Uniface.
            - Utilize exemplos de código dos documentos no contexto informado quando forem úteis.
                - Leia e interprete o contexto atentamente.
                - Priorize trechos mais relevantes para a pergunta.
                - Não invente informações fora do contexto.
                - Não faça suposições baseadas em conhecimento externo.
            - Responda somente ao que foi solicitado, sem pedir informações adicionais.
            - Caso as informações necessárias não estejam nos documentos, responda que não é possível responder por falta de conhecimento.
            - Sempre responda em português.
            - Formate corretamente blocos de código quando forem utilizados.

            Objetivo:
            Fornecer uma resposta clara, direta e precisa, baseada estritamente na base de conhecimento.

            Contexto: {context}
        """

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
    execution_rag.context = [
        {"source": doc.metadata.get("source"), "content": doc.page_content}
        for doc in response["context"]
    ]

    return execution_rag