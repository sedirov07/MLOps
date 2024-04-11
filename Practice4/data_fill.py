import pandas as pd


# Считываем датасет
df = pd.read_csv('datasets/titanic_filtered.csv')

df_filled = df.copy()
# Заполняем пропуски столбца Age алгебраическим средним
df_filled['Age'] = df_filled['Age'].fillna(df_filled['Age'].mean())

# Сохранение датасета
df_filled.to_csv('datasets/titanic_filtered.csv', index=False)
