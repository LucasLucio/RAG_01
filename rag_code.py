# Importando bibliotecas necessárias
import datetime
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import base_manager
from classes import ExecutionRag, FilesInRag
import model_questions

def pre_processing_question(question) -> FilesInRag:

    # Lista arquivos disponíveis
    files_rag = base_manager.list_docs_names("codes")

    files_in_rag = FilesInRag()
    files_in_rag.datetime_start = datetime.now()

    # Gera pseudo código para o questionamento
    pseudo_code = model_questions.execute_question(
    question,
    (
        """
            Você é um especialista em codificação.
            Responda sempre gerando pseudo-código, sem utilizar nenhuma linguagem específica.

            Regras:

                - Retorne somente o pseudo-código solicitado.
                - Não adicione explicações, comentários, títulos ou texto antes ou depois da resposta.
                - Não utilize sintaxe específica de linguagens reais; descreva a lógica de forma genérica.
                - Atenda exatamente ao pedido do usuário.
            
            Objetivo:
                - Fornecer pseudo-código claro, direto e representativo da solução solicitada.
        """
    ))

    files_in_rag.pseudo_code = pseudo_code
    files_in_rag.datetime_pseudo_code = datetime.now()

    # Prompt e ajuste de input para selecionar arquivos
    question_files = (
        f"Para o seguinte questionamento do usuário {question}, foi criado o seguinte "
        f"pseudo código: {pseudo_code}. Analise dentro destes arquivos disponíveis ({files_rag}) "
        "quais acredita conter exemplos de código em uniface que irão auxiliar "
        "em transcrever o código para a linguagem."
    )
    prompt_input = (

        """
            Você é um especialista em codificação em Uniface.
            Responda apenas e exatamente ao que for solicitado.

            Regras:
                - Retorne somente uma lista simples contendo 4 nomes de arquivos, separados por ponto e vírgula (;).
                - Não invente, modifique ou crie novos arquivos; utilize apenas os arquivos presentes no contexto.
                - Avalie o questionamento do usuário e inclua também arquivos que possam ser úteis para atender a solicitação.
                - Não adicione texto antes ou depois da lista (sem comentários, títulos ou explicações).

            Objetivo:
                - Fornecer uma lista direta, relevante e completa de arquivos conforme o pedido do usuário.
        """
    )

    # Define os arquivos que serão utilizados
    files_need = base_manager.define_files_need(
        question_files,
        prompt_input,
        files_rag
    )

    files_need = [f"files-input/codes/codes-{file}.pdf" for file in files_need]

    files_in_rag.files_defined = files_need

    files_in_rag.datetime_end = datetime.now()

    return files_in_rag


def rag_code(question) -> ExecutionRag:

    execution_rag = ExecutionRag()
    execution_rag.datetime_start = datetime.now()

    model = ChatOllama(model="llama3", temperature=0.7)

    vectorstore = base_manager.load_vectorstore()

    files_needed = pre_processing_question(question)

    execution_rag.files_used = files_needed

    filter_dict = {"source": {"$in": files_needed.files_defined}}

    # Recupera inicialmente k_max documentos
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10, 'filter': filter_dict},
    )

    # Cria prompt de sistema
    system_prompt = (

        """
            Você é um assistente de programação especializado em codificação Uniface.
            Responda sempre com base nos documentos fornecidos pelo RAG.

            Regras:
                - Crie o código solicitado em Uniface, usando como referência os exemplos e sintaxes presentes no contexto.
                - Atenda diretamente ao pedido do usuário, sem solicitar informações adicionais.
                - Se não houver informações suficientes no contexto para gerar o código, responda que não é possível.
                - Responda sempre em português.
                - Formate corretamente os blocos de código.
        
            Objetivo:
                - Fornecer código funcional e coerente com o exemplo dos documentos, atendendo exatamente ao pedido do usuário.
        
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

    execution_rag.datetime_end = datetime.now()

    return execution_rag