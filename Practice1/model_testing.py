import pickle
import numpy as np


# Загружаем модель
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Загружаем данные для тестирования
test_data = np.load("test/test_data_scaled.npy")

X_test = test_data[:, :-1]  # признаки до последнего
y_test = test_data[:, -1]   # последний признак

# Проверяем модель
score = model.score(X_test, y_test)
print("Test score: ", round(score, 2))
