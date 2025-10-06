# Importando bibliotecas necessárias
import execute

def main():
    question = input("Digite sua pergunta: ")
    resposta = execute.execute_question(question)
    print("Resposta:", resposta)

if __name__ == "__main__":
    main()
