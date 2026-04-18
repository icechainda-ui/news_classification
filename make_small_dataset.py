import pandas as pd

print("Начало...")

df = pd.read_csv("data/train_news.csv")

print("Файл загружен")

df_small = df.sample(n=5000, random_state=42)

print("Выборка сделана")

df_small.to_csv("data/train_small.csv", index=False)

print("Создан train_small.csv")