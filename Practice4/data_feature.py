import pandas as pd


# Считываем датасет
df = pd.read_csv('datasets/titanic_filtered.csv')

# Создание one-hot dataframe
encoded_sex = pd.get_dummies(df['Sex'])

# Замена Sex на one-hot столбцы
df = pd.concat([df.drop('Sex', axis=1), encoded_sex], axis=1)

# Сохранение датасета
df.to_csv('datasets/titanic_filtered.csv', index=False)
