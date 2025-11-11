# Importando bibliotecas necessárias
import datetime
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import base_manager
from classes import ExecutionRag, FilesInRag
import model_questions

def pre_processing_question(question):

    # Lista arquivos disponíveis
    files_rag = base_manager.list_docs_names("all")

    files_in_rag = FilesInRag()
    files_in_rag.datetime_start = datetime.now()

    # Gera pseudo código para o questionamento
    pseudo_code = model_questions.execute_question(
    question,
    (
        "Considere que você é um especialista em codificação. "
        "Responda sempre gerando um pseudo código, sem utilizar uma linguagem específica."
        "Não acrescente nada ao início nem ao final da resposta, apenas o pseudo código."
        "Não é necessário explicar o pseudo código, apenas apresente o pseudo código."
    ))

    files_in_rag.pseudo_code = pseudo_code
    files_in_rag.datetime_pseudo_code = datetime.now()
        
    # Prompt e ajuste de input para selecionar arquivos
    question_files = (
        f"Para o seguinte questionamento do usuário {question}, foi criado o seguinte "
        f"pseudo código: {pseudo_code}. Analise dentro destes arquivos disponíveis ({files_rag}) "
        "quais acredita conter exemplos de código em Uniface ou explicações da tecnologia que irão auxiliar "
        "em transcrever o código para a linguagem e na elaboração de uma resposta completa, concisa e correta."
    )
    prompt_input = (
        """
            Você é um especialista em codificação em Uniface e também um assistente de perguntas e respostas especializado na tecnologia Uniface.
            Responda apenas e exatamente ao que foi solicitado.

            Regras:

                - Retorne somente uma lista simples contendo nomes de arquivos separados por ponto e vírgula (;).
                - Não invente, modifique ou crie novos arquivos; utilize apenas os arquivos presentes no contexto.
                - Avalie o questionamento do usuário e inclua também arquivos que possam ser úteis para atender à solicitação.
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

    # Remove duplicatas da lista de arquivos necessários
    files_need = list(set(files_need))

    return files_need

def rag_both(question) ->ExecutionRag:

    execution_rag = ExecutionRag()
    execution_rag.datetime_start = datetime.datetime.now()

    model = ChatOllama(model="llama3", temperature=0.6)

    vectorstore = base_manager.load_vectorstore()

    files_needed = pre_processing_question(question)
    # Recupera inicialmente k_max documentos
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 15, "fetch_k": 20},
        filter=lambda doc: doc.metadata.get("source") in files_needed,
    )

    # Cria prompt de sistema
    system_prompt = (
        "Você é um assistente de perguntas e respostas especializado na tecnologia Uniface.\n"
        "Responda a solicitação do usuário com base nos documentos fornecidos.\n"
        "Utilize a base de conhecimento contida nos documentos para responder a pergunta do usuário.\n"
        "Se necessário criar códigos em Uniface, faça-o, usando como base os exemplos de códigos contidos nos documentos fornecidos, e demais exemplos de sintaxe também passadas.\n"
        "Mantenha o foco na tecnologia Uniface.\n"
        "Baseie suas respostas apenas sobre as informações contidas nos documentos.\n"
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
    
    return response["answer"]