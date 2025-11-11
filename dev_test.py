# Importando bibliotecas necessárias
import execute

def main():
    steps = ['judge', 'router', 'filter']
    question = input("Digite sua pergunta: ")
    resposta = execute.execute_question(question, steps)
    print(resposta.to_json())
    print("Resposta:", resposta.question)

if __name__ == "__main__":
    main()
