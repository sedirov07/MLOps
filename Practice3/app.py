import streamlit as st
from model import get_answer


# Отображение интерфейса Streamlit
st.title("Вопросно-ответный ассистент")

question = st.text_input("Введите ваш вопрос:")
context = st.text_area("Введите контекст:")

if st.button("Получить ответ"):
    if question and context:
        answer = get_answer(question, context)
        st.write("Ответ:", answer['answer'].capitalize())
    else:
        st.warning("Пожалуйста, введите вопрос и контекст.")
