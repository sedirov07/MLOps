from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Загрузка модели и токенизатора
model_name = "deepset/roberta-base-squad2"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Создание объекта pipeline для вопросно-ответного задания
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)


# Функция для ответа на вопрос
def get_answer(question, context):
    qa_input = {
        'question': question,
        'context': context
    }
    res = nlp(qa_input)
    return res
