# Importando bibliotecas necessárias
import router
import judge

def execute_question(question):
   
   valid_responses = False
   attempts = 0
   max_attempts = 3

   while not valid_responses and attempts < max_attempts:
    attempt_response = router.redirect_question(question)
    if attempt_response and isinstance(attempt_response, str) and len(attempt_response) > 0:
        validate_responses = judge.judge_answer(question, attempt_response, metric=0.7)
        if validate_responses.is_correct:
            return attempt_response
    else:
        attempts += 1
        continue
   return "Desculpe, não consegui gerar uma resposta adequada no momento. Por favor, tente novamente."