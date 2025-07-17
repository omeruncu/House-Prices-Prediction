import pandas as pd
import os


def load_data(config):
    """
    train.csv dosyasını 'Id' sütunu olmadan yükler.
    test.csv dosyasından 'Id' sütununu ayırarak data/processed/id_values.csv olarak kaydeder.

    Returns:
        train_df: 'Id'siz eğitim verisi
        test_df: 'Id'siz test verisi
    """
    # Yol tanımları
    train_path = config['data_paths']['train']
    test_path = config['data_paths']['test']
    id_save_path = config['data_paths']['test_ids']

    # Veri yükleme
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 'Id' sütunlarını çıkar
    train_df.drop(columns=["Id"], inplace=True)
    test_ids = test_df["Id"]
    test_df.drop(columns=["Id"], inplace=True)

    # Id sütununu kaydet
    os.makedirs(os.path.dirname(id_save_path), exist_ok=True)
    test_ids.to_frame().to_csv(id_save_path, index=False)
    print(f"✅ Test Id değerleri '{id_save_path}' konumuna kaydedildi.")

    return train_df, test_df
