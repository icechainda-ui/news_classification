import pandas as pd

def prepare_dataset(input_path, output_path, text_columns, category_column):
    df = pd.read_csv(input_path, low_memory=False)

    df[category_column] = df[category_column].astype(str)

    print("Колонки в датасете:", df.columns.tolist())

    # объединяем текстовые поля
    df["text"] = df[text_columns].fillna("").agg(" ".join, axis=1)

    # берём категорию
    df["category"] = df[category_column]

    # оставляем нужное
    df = df[["text", "category"]].dropna()

    df.to_csv(output_path, index=False)

    print(f"Готово! Создан {output_path}")


if __name__ == "__main__":
    prepare_dataset(
        input_path="data/lenta-ru-news.csv",
        output_path="data/train_news.csv",
        text_columns=["title", "text"],
        category_column="topic"
    )