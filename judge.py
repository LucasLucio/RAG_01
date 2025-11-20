# Importando bibliotecas necessárias
from datetime import datetime
from classes import JudgeResult
import model_questions


def judge_answer(question, answer, metric=0.7) -> JudgeResult:

    judge_result = JudgeResult()
    judge_result.datetime_start = datetime.now()
    judge_result.metric = metric

    evaluation = model_questions.execute_question(
        f"Pergunta: {question}\nResposta: {answer}\nA resposta está correta e completa?",
        (

            """
                Você é um especialista na tecnologia Uniface.
                Responda apenas e exatamente ao que for solicitado.

                Regras:
                    - Avalie se a resposta fornecida está correta e completa em relação à pergunta feita.
                    - Retorne somente um valor numérico entre 0.0 e 1.0, com até duas casas decimais.
                    - Utilize valores intermediários para indicar níveis de correção e completude.
                    - Não adicione texto antes ou depois do valor numérico.
                    - 
                
                Critérios de avaliação:
                    - A resposta deve ser tecnicamente correta.
                    - A resposta deve atender diretamente ao que foi solicitado.
                    - Se a resposta solicitar informações adicionais ao usuário, considere incorreta.
                    - Se a resposta fizer perguntas ao usuário, considere incorreta.
                    - Se a resposta não atender ao pedido original, considere incorreta.

                Objetivo:
                    - Classificar a precisão e a completude da resposta de forma objetiva e numérica.
            """
        ),
        temperature=0.7,
    )

    judge_result.datetime_end = datetime.now()

    try:
        evaluation = float(evaluation)
        judge_result.evaluation = evaluation
        judge_result.is_correct = True if evaluation >= metric else False
        judge_result.occurred_error = False

    except ValueError:
        judge_result.evaluation = None
        judge_result.is_correct = False
        judge_result.occurred_error = True
        judge_result.judge_response = evaluation

    return judge_result

      
