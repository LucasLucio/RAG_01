from log_database import start_database
import streamlit as st
import re
import execute

def email_valido(email: str) -> bool:
    padrao = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(padrao, email) is not None

##---------------------------

# Lógica principal da aplicação



steps = ['judge', 'router', 'filter']

st.set_page_config(page_title="Uniface IA", layout="centered", page_icon="\U0001F916")
  


st.markdown("<h2 style='text-align: center;'>Uniface IA Tool</h2>", unsafe_allow_html=True)
st.divider()

# Controle de estado
if "processando" not in st.session_state:
    st.session_state.processando = False

if "resposta" not in st.session_state:
    st.session_state.resposta = ""


# Identificação dos usuários
st.markdown("<h3 style='text-align: center;'>Identificação do Usuário</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Nome", max_chars=100)

with col2:
    email = st.text_input("E-mail", max_chars=100)

# Validação visual
email_ok = email_valido(email) if email else False

if email and not email_ok:
    st.warning("Digite um e-mail válido (exemplo: nome@dominio.com)")


position = st.selectbox(
    "Cargo",
    ["Selecione...", "Estagiário", "Trainee", "Júnior", "Pleno", "Sênior"]
)

# Validação de campos
dados_validos = (
    name.strip() != "" and
    email.strip() != "" and
    position != "Selecione..."
)

st.divider()

st.markdown("<h3 style='text-align: center;'>Faça sua pergunta</h3>", unsafe_allow_html=True)

pergunta = st.text_area(
    "Digite sua pergunta",
    disabled=not dados_validos or st.session_state.processando,
    max_chars=350,
    
)

botao_desabilitado = (not dados_validos) or st.session_state.processando

enviar = st.button(
    "\u2B9E",
    disabled=botao_desabilitado,
    type="secondary"
)


# Processamento de questionamento
if enviar and pergunta.strip() != "":
    st.session_state.processando = True
    st.session_state.resposta = ""

    # Ícone de carregamento
    with st.spinner("A IA está pensando..."):
        resposta = execute.execute_question(pergunta, steps, name, email, position)
        response = next((x for x in resposta.executions_rag if x.returned == True), None).response
        
        st.session_state.resposta = response

    st.session_state.processando = False

# Exibindo a resposta
if st.session_state.resposta:
    st.write("### Aqui está sua resposta:")
    st.success(st.session_state.resposta)