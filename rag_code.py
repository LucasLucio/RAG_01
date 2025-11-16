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
    files_rag = base_manager.list_docs_names("codes")

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
            Você é um especialista em codificação em Uniface.
            Responda apenas e exatamente ao que for solicitado.

            Regras:
                - Retorne somente uma lista simples contendo somente 4 nomes de arquivos, separados por ponto e vírgula (;).
                - Os arquivos devem ser ordenados do mais relevante ao menos relevante, considerando o que foi solicitado.
                - Não invente, não modifique e não crie novos arquivos. Utilize apenas os fornecidos como disponíveis.
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
        4
    )

    files_need = [f"files-input/codes/codes-{file}.txt" for file in files_need]

    files_in_rag.files_defined = files_need
    files_in_rag.files_supose = files_supose

    files_in_rag.datetime_end = datetime.now()

    return files_in_rag


def rag_code(question, steps) -> ExecutionRag:

    execution_rag = ExecutionRag()
    execution_rag.datetime_start = datetime.now()

    model = ChatOllama(model="llama3", temperature=0.7)

    vectorstore = base_manager.load_vectorstore()

    files_needed = pre_processing_question(question)

    execution_rag.files_used = files_needed

    retriever = base_manager.create_retriever(vectorstore, files_needed.files_defined, steps, 5)

    # Cria prompt de sistema
    system_prompt = (
       """
           Você é um assistente de programação especializado em codificação na tecnlogia Uniface.
           Gere os códigos solicitados em Uniface, utilizando os exemplos de código do contexto como documentação da linguagem.

            Regras:
               - Gere exclusivamente código Uniface.
               - Nunca gere código em outras linguagens como Python, JavaScript, Java ou pseudo-código.
               - Use como referência para a geração os exemplos de código presentes no contexto.
               - Atenda diretamente ao pedido do usuário, sem solicitar informações adicionais.
               - Se não houver informações suficientes no contexto para gerar o código, responda que não é possível.
               - Responda sempre em português.
               - Formate corretamente os blocos de código.

            Recomendações para interpretação do código Uniface:
               - Todo ; em Uniface é um comentário de linha, sempre após o ; deve haver o texto do comentário.
       
            Objetivo:
               - Fornecer código em Uniface funcional e coerente com o exemplo dos contextos, atendendo exatamente ao pedido do usuário.
           
            Aqui estão os documentos de contexto que você deve usar: 
           {context}
       """
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    # Cria a cadeia de documentos + modelo
    question_answer_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
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