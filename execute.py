# Importando bibliotecas necessárias
from dataclasses import asdict
from datetime import datetime
from classes import Executions
import router
import judge
import uuid
import log_database

def execute_question(question: str, steps: list, name: str, email: str, position: str) -> Executions:
    if 'judge' in steps:
        response = execute_judge(question, steps)
    else:
        response = execute_direct(question, steps)

    response.steps = steps
    response.name = name
    response.email = email
    response.position = position

    try:
        id = str(uuid.uuid4())
        response.id = id
        log_database.insert_log(id, asdict(response))
    except Exception as e:
        # Em caso de erro ao gerar UUID, logar o erro
        print(f"Erro ao gerar ID: {e}")

    return response


def execute_direct(question, steps: list) -> Executions:
    executions = Executions()
    executions.question = question
    executions.datetime_start = datetime.now()

    execution_rag = router.redirect(question, steps)
    execution_rag.attempt = True
    execution_rag.returned = False

    if execution_rag and isinstance(execution_rag.response, str) and len(execution_rag.response) > 0:
        execution_rag.returned = True
        executions.executions_rag.append(execution_rag)
        executions.default_return = False
        executions.datetime_end = datetime.now()
    else:
        executions.default_return = True
        executions.datetime_end = datetime.now()
        executions.response = "Desculpe, não consegui gerar uma resposta adequada no momento. Por favor, tente novamente."

    return executions


def execute_judge(question, steps: list) -> Executions:

   valid_responses = False
   attempts = 0
   max_attempts = 3

   executions = Executions()
   executions.question = question
   executions.datetime_start = datetime.now()

   while not valid_responses and attempts < max_attempts:
    execution_rag = router.redirect(question, steps)
    execution_rag.attempt = attempts + 1
    execution_rag.returned = False
    if execution_rag and isinstance(execution_rag.response, str) and len(execution_rag.response) > 0:
        validate_responses = judge.judge_answer(question, execution_rag.response, metric=0.7)
        execution_rag.judge_result = validate_responses
        if validate_responses.is_correct == True:
            executions.executions_rag.append(execution_rag)
            break
        else:
            executions.executions_rag.append(execution_rag)
            attempts += 1
            continue
    else:
        executions.executions_rag.append(execution_rag)
        attempts += 1
        continue
    
   if len(executions.executions_rag) <= 0:
       executions.default_return = True
       executions.datetime_end = datetime.now()
       executions.response = "Desculpe, não consegui gerar uma resposta adequada no momento. Por favor, tente novamente."
   else:
        best_evaluation = 0.0
        best_execution = None
        for id in range(len(executions.executions_rag)):
            execution = executions.executions_rag[id]
            if execution.judge_result and execution.judge_result.evaluation:
                if execution.judge_result.evaluation > best_evaluation:
                    best_evaluation = execution.judge_result.evaluation
                    best_execution = id
        
        if best_execution is not None:
            executions.executions_rag[best_execution].returned = True
            executions.default_return = False
            executions.datetime_end = datetime.now()

   return executions