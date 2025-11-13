from datetime import datetime
from classes import ExecutionRag, Redirect
# Importando bibliotecas necessárias
import model_questions
import rag_docs
import rag_code
import rag_both

def redirect(question, steps: list) -> ExecutionRag:
    if 'router' in steps:
        return execute_router(question, steps)
    else:
        return execute_direct(question, steps)



def define_type_question(question) -> Redirect:
    datetime_start = datetime.now()
    type_question = model_questions.execute_question(
        question,
        (
            """
                Você é um especialista em codificação.
                Responda apenas e exatamente ao que foi solicitado.

                Regras:
                    - Classifique o questionamento do usuário em código, conceito ou ambos, considerando quais termos das diretrizes de interpretação aparecem no questionamento.
                    - Sua resposta deve conter apenas uma dessas três palavras, sem explicações adicionais.

                Diretrizes de interpretação:

                    - Termos no questionamento como: “faça algo”, “como faço para”, “gere um código que”, “crie um script que”, “faça um programa que” -> tendem a ser código.
                    - Termos no questionamento como: “o que é”, “explique”, “defina”, “qual a diferença entre”, “quais são as vantagens de” -> tendem a ser conceito.
                    - Termos no questionamento como: “como funciona”, “me mostre um exemplo de”, “quais são os passos para”, “me ajude a entender” -> tendem a ser ambos.

                Objetivo:
                    - Classificar exatamente o que foi perguntado, considerando o retorno desejado pelo usuário.
            """
        ),
        temperature=0.7,
    )
    types_question_valid = ["código", "conceito", "ambos"]

    if type_question not in types_question_valid:
        final_type = "ambos"
    else:
        final_type = type_question

    datetime_end = datetime.now()
    return Redirect(datetime_start, datetime_end, final_type, type_question)


def execute_router(question, steps: list) -> ExecutionRag:
    define_type = define_type_question(question)
    type_question = define_type.final_type
    if type_question == "conceito":
        response = rag_docs.rag_docs(question, steps)
    elif type_question == "código":
        response = rag_code.rag_code(question, steps)
    else:  # ambos
        response = rag_both.rag_both(question, steps)

    response.type_question = define_type

    return response

def execute_direct(question, steps: list) -> ExecutionRag:
    return rag_both.rag_both(question, steps)
