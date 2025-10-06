# Importando bibliotecas necessárias
from langchain_ollama import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import base_manager
import model_questions

def pre_processing_question(question):

    # Lista arquivos disponíveis
    files_in_rag = base_manager.list_docs_names("codes")

    # Gera pseudo código para o questionamento
    pseudo_code = model_questions.execute_question(
    question,
    (
        "Considere que você é um especialista em codificação. "
        "Responda sempre gerando um pseudo código, sem utilizar uma linguagem específica."
        "Não acrescente nada ao início nem ao final da resposta, apenas o pseudo código."
        "Não é necessário explicar o pseudo código, apenas apresente o pseudo código."
    ))

    # Prompt e ajuste de input para selecionar arquivos
    question_files = (
        f"Para o seguinte questionamento do usuário {question}, foi criado o seguinte "
        f"pseudo código: {pseudo_code}. Analise dentro destes arquivos disponíveis ({files_in_rag}) "
        "quais acredita conter exemplos de código em uniface que irão auxiliar "
        "em transcrever o código para a linguagem."
    )
    prompt_input = (
        "Considere que você é um especialista em codificação em Uniface."
        "Responda apenas e exatamente o solicitado"
        "Responda apresentando somente uma lista simples com os nomes de arquivos separados por ;"
        "Não crie novos arquivos, utilize somente os apresentados"
        "Não acrescente nada ao início nem ao final da resposta, apenas os nomes dos arquivos."
        "Considere o questionamento do usuário para analisar se também há algum arquivo que possa ser útil, e inclua na resposta."
    )

    # Define os arquivos que serão utilizados
    files_need = base_manager.define_files_need(
        question_files,
        prompt_input,
        files_in_rag
    )

    return files_need


def rag_code(question):

    model = ChatOllama(model="llama3", temperature=0.7)

    vectorstore = base_manager.load_vectorstore()

    files_needed = pre_processing_question(question)
    # Recupera inicialmente k_max documentos
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 15},
        filter=lambda doc: doc.metadata.get("source") in files_needed,
    )

    # Cria prompt de sistema
    system_prompt = (
        "Você é um assistente de programação especializado em codificação Uniface.\n"
        "Responda a solicitação do usuário com base nos documentos fornecidos.\n"
        "Crie os códigos solicitados em Uniface, usando como base os exemplos de códigos contidos nos documentos fornecidos, e demais exemplos de sintaxe também passadas.\n"
        "Se não for possível realizar a criação, diga que não é possível.\n"
        "Sempre responda em português e formate os blocos de código corretamente.\n\n"
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
