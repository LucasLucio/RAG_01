#Importando bibliotecas necessárias
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama import ChatOllama
import os
from langchain_core.prompts import ChatPromptTemplate
import  nltk

#Baixando pacotes específicos do NLTK (Natural Language Toolkit)
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def simple_rag(directory, question):
    # Inicializa o modelo de chat Ollama com o modelo "llama3"
    model = ChatOllama(model="llama3", )
    # Define o diretório de persistência para o banco de dados Chroma
    persistance_directory = "./chroma_db"

    if directory is not None:
        # Carrega documentos do diretório especificado
        docLoad = DirectoryLoader(directory, glob="**/*.pdf", show_progress=True).load()
        # Divide os documentos em pedaços menores
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(docLoad)

        if os.path.exists(persistance_directory) and os.path.isdir(persistance_directory):
            # Carrega o banco de dados Chroma existente
            vectorstore = Chroma(persist_directory=persistance_directory, embedding_function=OllamaEmbeddings(model="llama3"))
        else:
            # Cria um novo banco de dados Chroma a partir dos documentos divididos
            vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model="llama3"), persist_directory=persistance_directory)
    else:
        if os.path.exists(persistance_directory) and os.path.isdir(persistance_directory):
            # Carrega o banco de dados Chroma existente
            vectorstore = Chroma(persist_directory=persistance_directory, embedding_function=OllamaEmbeddings(model="llama3"))
        else:
            # Imprime uma mensagem de erro se nenhum diretório for fornecido e o banco de dados persistente não existir
            print("Nenhum diretório fornecido e a base de dados persistente não existe.")
            return

    # Configura o recuperador de documentos com base na similaridade
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 30, "fetch_k": 50})

    # Define o prompt do sistema para o modelo de chat
    system_prompt = (
        "Considere que você é um assistente de programação com foco em codificação para Uniface."
        "Use os seguintes pedaços de contexto recuperado para responder as solicitações realizadas."
        "Se não souber as respostas ou não conseguir gerar algum código para a solicitação diga que não é possível realizar a operação desejada."
        "\n\n"
        "{context}"
    )

    # Cria o template de prompt para o modelo de chat
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # Cria a cadeia de perguntas e respostas usando o modelo e o prompt
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    # Cria a cadeia de recuperação usando o recuperador e a cadeia de perguntas e respostas
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Invoca a cadeia de recuperação com a pergunta fornecida
    response = rag_chain.invoke({"input": question})
    response["answer"]

    # Imprime a resposta obtida
    print("================== RESPONSE ==================")
    print(response["answer"])
    print("================ END RESPONSE ================")

def main():
    # Solicita ao usuário se deseja carregar os documentos de análise
    loadDocs = input("Deseja carregar os documentos de análise? (s/n): \n")

    # Verifica se o usuário deseja carregar os documentos
    if(loadDocs == 's'):
        # Solicita ao usuário o caminho para os documentos de análise
        directory = input("Caminho para os docs de análise: \n")
    else:
        # Define o diretório como None se o usuário não quiser carregar os documentos
        directory = None
    
    # Solicita ao usuário que digite sua pergunta
    question = """"
    Em Uniface utilizando ProcScript como linguagem de programação, crie uma operação que valide um CPF. 
   Deve receber uma string como parâmetro de entrada contendo o CPF sem formatação e é necessário que a operação retorne um booleano indicando se é um CPF válido. 
   Sendo os valores possíveis, verdadeiro para um CPF válido e falso para um inválido.

   As regras de validação de um CPF são as descritas abaixo:

   Para se validar um CPF se calcula o primeiro dígito verificador a partir dos 9 primeiros dígitos do CPF.
   Em seguida, calcula o segundo dígito verificador a partir dos 10 primeiros dígitos do CPF incluindo o primeiro dígito, obtido na primeira parte.

   Como calcular o primeiro dígito:

      Separar os primeiros 9 dígitos do CPF e multiplicar cada um dos números, 
      desde o primeiro até o nono número, inciar a multiplicação do primeiro em 10 e decrescer em 1 o multiplicador a cada dígito seguinte. 
      Somar todos os resultados obtidos nas multiplicações e dividir por 11 e considerar como quociente apenas o valor inteiro.
      Considere o resto da divisão para obter o resultado do dígito. 
         - Se for menor que 2, então o dígito é igual a 0 (Zero).
         - Se for maior ou igual a 2, então o dígito é igual a 11 menos o próprio resto da divisão.

   Como calcular o segundo dígito:

   Para  calcular o segundo dígito vamos usar o primeiro digito já calculado. 
   Vamos montar a mesma tabela de multiplicação usada no cálculo do primeiro dígito. 
   Só que desta vez usaremos na segunda linha os valores 11,10,9,8,7,6,5,4,3,2 já que estamos incluindo mais um digito no cálculo(o primeiro dígito calculado):

      O calculo é semelhante ao do primeiro dígito, a diferença neste caso é que são 10 dígitos a serem multiplicados, incluindo o primeiro verificador. 
      Para multiplicar se inicia em 11 como multiplicador do primeiro número e segue decrescendo até o décimo número. 
      Somar todos os resultados obtidos nas multiplicações e dividir por 11 e considerar como quociente apenas o valor inteiro.
      Considere o resto da divisão para obter o resultado do dígito. 
         - Se for menor que 2, então o dígito é igual a 0 (Zero).
         - Se for maior ou igual a 2, então o dígito é igual a 11 menos o próprio resto da divisão.

   após obter os dois últimos dígitos é necessário verificar se eles são iguais ao informado no CPF de entrada.
    """

    model = ChatOllama(model="llama3")

    resposta = model.invoke("Defina em uma lista simples com nada além dos itens definidos (contendo apenas o nome do item) quais estruturas e conceitos de linguagem de programação será necessário para responder a seguinte solicitação: \n" + question)

    question = question + " utilizando as seguintes estruturas e conceitos de linguagem de programação: \n" + resposta.content

    #input("Digite sua pergunta: \n")

    # Chama a função simple_rag passando o diretório e a pergunta como argumentos
    simple_rag(directory, question)


# Verifica se o script está sendo executado diretamente (não importado como módulo)
if __name__ == "__main__":
    # Chama a função main para iniciar o programa
    main()