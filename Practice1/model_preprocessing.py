from sklearn.preprocessing import StandardScaler
import numpy as np

# Загружаем данные
train_data = np.load("train/train_data.npy")
test_data = np.load("test/test_data.npy")

# Инициализируем и применяем StandardScaler
scalerTrain = StandardScaler()
scalerTrain.fit(train_data)

scalerTest = StandardScaler()
scalerTest.fit(test_data)

# Преобразуем данные и сохраняем
train_data_scaled = scalerTrain.transform(train_data)
test_data_scaled = scalerTest.transform(test_data)

np.save("train/train_data_scaled.npy", train_data_scaled)
np.save("test/test_data_scaled.npy", test_data_scaled)
