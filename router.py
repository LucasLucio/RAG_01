from datetime import datetime
from classes import ExecutionRag, Redirect
# Importando bibliotecas necessárias
import model_questions
import rag_docs
import rag_code
import rag_both


def define_type_question(question) -> Redirect:
    datetime_start = datetime.now()
    type_question = model_questions.execute_question(
        question,
        (
            """
                Você é um especialista em codificação.
                Responda apenas e exatamente ao que foi solicitado.

                Regras:
                    - Classifique o questionamento do usuário em código, conceito ou ambos.
                    - Sua resposta deve conter apenas uma dessas três palavras, sem explicações adicionais.
                    - Avalie se o usuário deseja a criação/geração de código ou se deseja entender um ou mais conceitos.
                    - Caso o questionamento envolva os dois, responda ambos.
                    - Se for um questionamento conceitual e você considerar que exemplos de código são necessários para explicar, responda ambos.
                    - O desejo do usuário pode ou não estar explícito na pergunta; avalie o contexto.

                Diretrizes de interpretação:

                    - Termos como: “faça algo”, “como faço para”, “gere um código que”, “crie um script que”, “faça um programa que” -> tendem a ser código.
                    - Termos como: “o que é”, “explique”, “defina”, “qual a diferença entre”, “quais são as vantagens de” -> tendem a ser conceito.
                    - Termos como: “como funciona”, “me mostre um exemplo de”, “quais são os passos para”, “me ajude a entender” -> podem ser ambos, dependendo do contexto.

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


def redirect_question(question) ->ExecutionRag:
    define_type = define_type_question(question)
    type_question = define_type.final_type
    if type_question == "conceito":
        response = rag_docs.rag_docs(question)
    elif type_question == "código":
        response = rag_code.rag_code(question)
    else:  # ambos
        response = rag_both.rag_both(question)

    response.type_question = define_type

    return response
