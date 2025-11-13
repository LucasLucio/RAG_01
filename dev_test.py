# Importando bibliotecas necessárias
import execute

def main():
    steps = ['judge', 'router', 'filter']
    #question = input("Digite sua pergunta: ")
    question = "Crie uma operation que some dois numeros"
    resposta = execute.execute_question(question, steps)
    print(resposta.to_json())

    response = next((x for x in resposta.executions_rag if x.returned == True), None).response
    print("Resposta:", response)

if __name__ == "__main__":
    main()
