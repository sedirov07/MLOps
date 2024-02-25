import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Загружаем предобработанные данные
train_data = np.load("train/train_data_scaled.npy")

# Создаем и обучаем модель
model = LinearRegression()

X_train = train_data[:, :-1]  # признаки до последнего
y_train = train_data[:, -1]   # последний признак
model.fit(X_train, y_train)

# Сохраняем обученную модель в файл с помощью pickle
pkl_filename = "model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)
