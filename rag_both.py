# Importando bibliotecas necessárias
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import base_manager
from classes import ExecutionRag, FilesInRag
import model_questions

def pre_processing_question(question) -> FilesInRag:

    # Lista arquivos disponíveis
    files_rag = base_manager.list_docs_names("all")

    files_in_rag = FilesInRag()
    files_in_rag.datetime_start = datetime.now()
    files_in_rag.files_available = [file for file in files_rag.split(",")]

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
            Você é um especialista em codificação em Uniface e também um assistente de perguntas e respostas especializado na tecnologia Uniface.
            Responda apenas e exatamente ao que foi solicitado.

            Regras:

                - Retorne somente uma lista simples contendo somente 6 nomes de arquivos, separados por ponto e vírgula (;).
                - Os arquivos devem ser ordenados do mais relevante ao menos relevante, considerando o que foi solicitado.
                - Não invente, não modifique e não crie novos arquivos. Utilize apenas os fornecidos como disponíveis.
                - Avalie o questionamento do usuário e inclua também arquivos que possam ser úteis para atender à solicitação.
                - Não adicione texto antes ou depois da lista (sem comentários, títulos ou explicações).

            Objetivo:
                - Fornecer uma lista direta, relevante e completa de arquivos conforme o pedido do usuário.
        """
    )

    # Define os arquivos que serão utilizados
    files_need, files_supose = base_manager.define_files_need(
        question_files,
        prompt_input,
        files_rag,
        6
    )

    # Remove duplicatas da lista de arquivos necessários
    files_need = list(set(files_need))

    #Filtra e define arquivos de código e documentos
    files_rag_docs = base_manager.list_docs_names("docs")
    files_rag_code = base_manager.list_docs_names("codes")

    files_need_docs = [file for file in files_need if file in files_rag_docs]
    files_need_docs_dir = [f"files-input/docs/docs-{file}.pdf" for file in files_need_docs]

    files_need_code = [file for file in files_need if file in files_rag_code]
    files_need_code_dir = [f"files-input/codes/codes-{file}.txt" for file in files_need_code]

    files_need = files_need_docs_dir + files_need_code_dir

    files_in_rag.files_defined = files_need
    files_in_rag.files_supose = files_supose

    files_in_rag.datetime_end = datetime.now()

    return files_in_rag

def rag_both(question, steps) -> ExecutionRag:

    execution_rag = ExecutionRag()
    execution_rag.datetime_start = datetime.now()

    model = ChatOllama(model="llama3", temperature=0.7)

    vectorstore = base_manager.load_vectorstore()

    files_needed = pre_processing_question(question)

    execution_rag.files_used = files_needed

    retriever = base_manager.create_retriever(vectorstore, files_needed.files_defined, steps)

    # Cria prompt de sistema
    system_prompt = (
        """
            Você é um assistente de perguntas e respostas especializado na tecnologia Uniface.
            Responda sempre com base nos documentos fornecidos pelo RAG.

            Regras:
                - Utilize exclusivamente a base de conhecimento presente nos documentos do contexto para responder à solicitação.
                - Se for necessário criar código em Uniface, use como referência os exemplos e sintaxes presentes no contexto.
                - Mantenha o foco apenas na tecnologia Uniface.
                - Não utilize conhecimento externo ou crie informações que não estejam documentadas.
                - Atenda diretamente ao que foi solicitado, sem pedir informações adicionais.
                - Se não houver informações suficientes para responder com assertividade, diga que não é possível responder por falta de conhecimento no contexto.
                - Responda sempre em português.
                - Formate corretamente quaisquer blocos de código fornecidos.

            
            Recomendações para interpretação do código Uniface:
                - Todo ; em Uniface é um comentário de linha, sempre após o ; deve haver o texto do comentário.
                
            Objetivo:
                - Fornecer uma resposta clara, assertiva e fundamentada apenas nos documentos disponíveis.
            
            Aqui estão os documentos de contexto que você deve usar: 
            {context}
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