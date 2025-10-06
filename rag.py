# Importando bibliotecas necessárias
import streamlit as st
import execute

# ===== Streamlit =====
st.title("Assistente de Programação Uniface")

txt_input = st.text_input("Como posso te ajudar? :")

if st.button("Consultar"):
    if txt_input:
        with st.spinner('Realizado operação, aguarde...'):
            resposta = execute(txt_input)
        st.success(resposta)
    else:
        st.warning("Por favor, digite uma pergunta.")
