import numpy as np
import os

# Создаем папки "train" и "test", если их еще нет
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Генерируем и сохраняем данные для тренировки
train_data = np.random.randn(70, 9)  # Пример случайных данных
np.save("train/train_data.npy", train_data)

# Генерируем и сохраняем данные для тестирования
test_data = np.random.randn(30, 9)  # Пример случайных данных
np.save("test/test_data.npy", test_data)
