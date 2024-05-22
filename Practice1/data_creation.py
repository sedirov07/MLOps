import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Создаем папки "train" и "test", если их еще нет
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Загружаем данные из CSV файла
data = pd.read_csv("laptop_price.csv", encoding="latin1")

columns_to_drop = ['laptop_ID', 'Weight', 'Product']
data.drop(columns=columns_to_drop, inplace=True)

# Преобразование категориальных признаков с помощью One-Hot Encoding
categorical_columns = data.select_dtypes(include=['object']).columns
data_encoded = pd.get_dummies(data, columns=categorical_columns)
data_encoded.replace({True: 1, False: 0}, inplace=True)

# Разделяем данные на обучающий и тестовый наборы в соотношении 70/30
train_data, test_data = train_test_split(data_encoded, test_size=0.3,
                                         random_state=42)

# Сохраняем данные для тренировки и тестирования
np.save("train/train_data.npy", train_data)
np.save("test/test_data.npy", test_data)
