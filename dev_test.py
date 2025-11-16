# Importando bibliotecas necessárias
import execute

def main():
    steps = ['judge', 'router', 'filter']
    #question = input("Digite sua pergunta: ")
    question = "Crie uma operation que seja semelhante a uma calculadora, deve receber dois números e a operação desejada, então retorne o resultado com base na operação realizada em Uniface."
    resposta = execute.execute_question(question, steps)
    print(resposta.to_json())

    response = next((x for x in resposta.executions_rag if x.returned == True), None).response
    print("Resposta:", response)

if __name__ == "__main__":
    main()
