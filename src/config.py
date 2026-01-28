class Config:
    model_name = "bert-base-uncased"
    max_length = 256
    train_batch_size = 8
    eval_batch_size = 8
    epochs = 3
    learning_rate = 2e-5

    train_file = "data/raw/train.csv"
    test_file = "data/raw/test.csv"

    save_dir = "models/best_model"
