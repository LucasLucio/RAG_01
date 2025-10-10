from classes import ExecutionRag, Redirect
# Importando bibliotecas necessárias
import model_questions
import rag_docs
import rag_code
import rag_both


def define_type_question(question) -> Redirect:
    type_question = model_questions.execute_question(
        question,
        (
            "Considere que você é um especialista em codificação. "
            "Responda apenas e exatamente o solicitado"
            "Avalie se o questionamento é relacionado a uma solicitação de criação e geração de código, ou se é um questionamento no qual o usuário quer entender um ou mais conceitos. Caso acredite ser um questionamento que envolva os dois, responda que é ambos."
            "Caso seja um questionamento para entender um conceito e acredite ser necessário apresentar exemplos de código para auxiliar na explicação, considere como ambos."
            "A sua saída deve ser apenas uma palavra, que pode ser: código, conceito ou ambos, dependendo da sua avaliação, conforme os critérios definidos acima."
            "Considere que pra a avaliação do tipo de questionamento, o desejo do usuário pode ou não estar explícito na pergunta."
            "Questionamentos que incluam termos como 'faça algo', 'como faço para', 'gere um código que', 'crie um script que', 'faça um programa que', ou similares, tendem a serem mais classificados como código."
            "Questionamentos que incluam termos como 'o que é', 'explique', 'defina', 'qual a diferença entre', 'quais são as vantagens de', ou similares, tendem a serem mais classificados como conceito."
            "Questionamentos que incluam termos como 'como funciona', 'me mostre um exemplo de', 'quais são os passos para', 'me ajude a entender', ou similares, podem ser classificadas como ambos, dependendo do contexto."
            "Avalie exatamente o que foi perguntado, e responda apenas com uma dessas três palavras. Considerando o que o usuário deseja ter como retorno."
        ),
        temperature=0.3,
    )
    types_question_valid = ["código", "conceito", "ambos"]

    if type_question not in types_question_valid:
        final_type = "ambos"
    else:
        final_type = type_question

    return Redirect(final_type, type_question)


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
