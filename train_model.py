from ml_core import train_model

if __name__ == "__main__":
    acc = train_model("data/train_news.csv")
    print(f"Обучение закончено. Accuracy: {acc:.2f}")