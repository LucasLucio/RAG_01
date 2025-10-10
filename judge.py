# Importando bibliotecas necessárias
import model_questions


def judge_answer(question, answer, metric=0.7):
    evaluation = model_questions.execute_question(
        f"Pergunta: {question}\nResposta: {answer}\nA resposta está correta e completa?",
        (
            "Considere que você é um especialista em codificação Uniface."
            "Responda apenas e exatamente o solicitado"
            "Avalie se a resposta fornecida está correta e completa para a pergunta feita."
            "A sua saída deve ser apenas valor numérico entre 0 e 1, podendo variar de 0.0 a 1.0, onde 0.0 significa que a resposta está totalmente incorreta, e 1.0 significa que a resposta está totalmente correta."
            "As variações intermediárias devem ser utilizadas para indicar o quão correta e completa está a resposta."
            "As variações intermediárias podem utilizar de qualquer valor decimal entre 0.0 e 1.0, como por exemplo: 0.2, 0.35, 0.68, 0.79, etc. Não passando de duas casas decimais após o ponto."
            "Não acrescente nada ao início nem ao final da resposta, apenas o valor numérico."
            "Durante a avaliação, considere que a resposta deve ser tecnicamente correta, e também deve atender ao que foi solicitado na pergunta."
            "Caso a resposta não esteja atendendo diretamente ao que foi solicitado, considere como incorreta."
            "Se a resposta for uma nova solicitação ao usuário para complementar a solicitação original, considere como incorreta."
            "Se a resposta for um questionamento direcionado ao usuário responder, considere como incorreta."
        ),
        temperature=0.5,
    )
    try:
        evaluation = float(evaluation)
        return {
            "evaluation": evaluation,
            "is_correct": True if evaluation >= metric else False,
            "occurred_error": False
        }
    except ValueError:
        return {
            "evaluation": None,
            "is_correct": False,
            "occurred_error": True,
            "judge_response": evaluation
        }
