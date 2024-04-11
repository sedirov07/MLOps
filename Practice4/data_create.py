from catboost.datasets import titanic

# Создаем датасет
train_df, _ = titanic()

# Отфильтровываем необходимые признаки (класс, пол, возраст)
filtered_df = train_df[['Pclass', 'Sex', 'Age']]

# Сохраняем DataFrame в CSV файл
train_df.to_csv('datasets/titanic.csv', index=False)
filtered_df.to_csv('datasets/titanic_filtered.csv', index=False)
