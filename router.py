# Importando bibliotecas necessárias
import model_questions
import rag_docs
import rag_code
import rag_both


def define_type_question(question):
    type_question = model_questions.execute_question(
        question,
        (
            "Considere que você é um especialista em codificação. "
            "Responda apenas e exatamente o solicitado"
            "Avalie se o questionamento é relacionado a criação de código, ou se é um questionamento sobre conceitos. Caso acredite ser um questionamento que envolva os dois, responda que é ambos."
            "Caso acredite ser necessário apresentar exemplos de código para auxiliar na explicação, considere como ambos."
            "A sua saída deve ser apenas uma palavra, que pode ser: código, conceito ou ambos."
            "Não acrescente nada ao início nem ao final da resposta, apenas os nomes dos arquivos."
            "Avalie exatamente o que foi perguntado, e responda apenas com uma dessas três palavras. Considerando o que o usuário deseja ter como retorno."
            "Considere que a sua resposta será utilizada para definir os exemplos que serão apresentados para auxiliar a responder o questionamento de forma completa posteriormente"
        ),
        temperature=0.3,
    )
    types_question_valid = ["código", "conceito", "ambos"]

    if type_question not in types_question_valid:
        return "ambos"
    else:
        return type_question

def redirect_question(question):
    type_question = define_type_question(question)
    if type_question == "conceito":
       response = rag_docs.rag_docs(question)
    elif type_question == "código":
       response = rag_code.rag_code(question)
    else:  # ambos
       response = rag_both.rag_both(question)

    return response
