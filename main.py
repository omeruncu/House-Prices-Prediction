import yaml
from src.data_loader import load_data
from src.preprocess import EdaPreprocessor


if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_df, test_df = load_data(config)

    numeric_in_cat = [
        "OverallQual","OverallCond","BsmtFullBath","BsmtHalfBath",
        "FullBath","HalfBath","BedroomAbvGr","TotRmsAbvGrd",
        "Fireplaces","GarageCars","MoSold","YrSold"
    ]

    processor = EdaPreprocessor(
        train_df,
        test_df,
        config_path="config/config.yaml",
        numeric_in_cat=numeric_in_cat
    )

    combined_df, cat_cols, num_cols = processor.run()
