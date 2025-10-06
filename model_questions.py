# Importando bibliotecas necessárias
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage


def execute_question(question, prompt, temperature=0.7):
    model = ChatOllama(model="llama3", temperature=temperature)

    # Mensagem de sistema que define o prompt específico
    system_prompt = SystemMessage(content=prompt)

    # Pergunta do usuário
    user_question = HumanMessage(content=question)

    # Chamada do modelo com o prompt agregado
    response = model.invoke([system_prompt, user_question])  # Passa lista de mensagens

    return response.content  # .content contém a resposta

def main():
    execute_question("Crie uma operation que some dois numeros", "Crie uma operação em Uniface que some dois números.")


if __name__ == "__main__":
    main()
