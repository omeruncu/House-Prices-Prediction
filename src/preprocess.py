import yaml
import pandas as pd
import os
import pickle

import src.utils.eda_helpers as hlp
from .utils.drop_tracker import DropTracker


class EdaPreprocessor:
    def __init__(self,
                 train_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 config_path: str,
                 numeric_in_cat: list):
        # Config yolu ve data_paths
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.paths = cfg["data_paths"]

        # DropTracker kaydı için
        self.tracker = DropTracker(save_path=cfg["data_paths"]["drop_columns"])

        # Train/test birleşimi
        self.df = pd.concat([train_df, test_df], ignore_index=True)
        self.df.reset_index(drop=True, inplace=True)

        # Kategorik/nümerik sütun listeleri
        self.cat_cols, self.num_cols, self.cat_but_car = hlp.grab_col_names(self.df)
        self.numeric_in_cat = numeric_in_cat

    def drop_dominance(self):
        drop_cands, self.dominance_table = hlp.dominant_class_filter(self.df, self.cat_cols)
        self.df = self.tracker.drop(self.df, drop_cands, stage_name="dominance_filter")
        return self

    def fill_structural(self):
        if isinstance(self.df, tuple):
            self.df = self.df[0]
        df, changed_cols = hlp.fill_structural_nans(self.df)
        self.df = df
        self.structural_cols = changed_cols
        return self

    def fill_saleprice(self):
        # NaN olan test seti SalePrice sütununu 0 ile doldur
        self.df.loc[self.df["SalePrice"].isna(), "SalePrice"] = 0
        return self

    def regrab_columns(self):
        # Dominance + structural fill sonrası kolon tiplerini yeniden çıkar
        self.cat_cols, self.num_cols, self.cat_but_car = hlp.grab_col_names(self.df)
        return self

    def reclassify_columns(self):
        # numeric_in_cat öğelerini cat→num taşı ve kalan cat_cols'u object yap
        self.df, self.cat_cols, self.num_cols = hlp.reclassify_columns(
            self.df,
            self.cat_cols,
            self.num_cols,
            self.numeric_in_cat
        )
        return self

    def simple_impute(self):
        self.df = hlp.simple_imputer(self.df, self.num_cols, self.cat_cols)
        return self

    def rare_encode(self, rare_th=0.01):
        self.df = hlp.rare_encoder(self.df, self.cat_cols, rare_th)
        return self

    def drop_saleprice_from_num(self):
        # Model girdi setinde SalePrice bir label olduğundan num_cols'tan çıkart
        self.num_cols = [c for c in self.num_cols if c != "SalePrice"]
        return self

    def cap_outliers(self):
        # replace_with_thresholds inplace çalıştığı için return etmiyoruz
        for col in self.num_cols:
            hlp.replace_with_thresholds(self.df, col)
        return self

    def drop_zero_variance(self):
        self.df, self.num_cols, drop_cols = hlp.drop_zero_mean(self.df, self.num_cols)
        self.tracker.history["zero_variance"] = drop_cols
        return self

    def finalize(self):
        """
                - Bütün silinen sütunları tek bir listede toplar ve kaydeder
                - Güncel num_cols ve cat_cols listelerini kaydeder
                - num_cols için min-max değerleri, cat_cols için unique değerleri kaydeder
                """
        # 1) Tüm drop aşamalarını tek bir listede topla
        all_drops = []
        for cols in self.tracker.history.values():
            all_drops.extend(cols)
        # Aynı kolonu birden fazla kaydetmemek için unique et
        all_drops = list(dict.fromkeys(all_drops))

        # 2) Dosya yollarını config’den al
        paths = self.paths
        save_paths = [
            paths["drop_columns"],
            paths["num_cols"],
            paths["cat_cols"],
            paths["num_cols_min_max"],
            paths["cat_cols_values"],
        ]
        # Klasörleri hazırla
        for p in save_paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)

        # 3) drop_columns.pkl
        with open(paths["drop_columns"], "wb") as f:
            pickle.dump(all_drops, f)

        # 4) num_cols.pkl
        with open(paths["num_cols"], "wb") as f:
            pickle.dump(self.num_cols, f)

        # 5) cat_cols.pkl
        with open(paths["cat_cols"], "wb") as f:
            pickle.dump(self.cat_cols, f)

        # 6) num_cols_min_max.pkl
        num_min_max = {
            col: (self.df[col].min(), self.df[col].max())
            for col in self.num_cols
        }
        with open(paths["num_cols_min_max"], "wb") as f:
            pickle.dump(num_min_max, f)

        # 7) cat_cols_values.pkl
        cat_values = {
            col: self.df[col].dropna().unique().tolist()
            for col in self.cat_cols
        }
        with open(paths["cat_cols_values"], "wb") as f:
            pickle.dump(cat_values, f)

        # 8) İşlenmiş combined df’i CSV olarak da kaydet
        proc_dir = os.path.dirname(paths["drop_columns"])
        self.df.to_csv(f"{proc_dir}/combined_processed.csv", index=False)

        # Son olarak df’i, cat_cols ve num_cols’u döndür
        return self.df, self.cat_cols, self.num_cols


    def run(self):
        return (
            self
            .drop_dominance()
            .fill_structural()
            .fill_saleprice()
            .regrab_columns()
            .reclassify_columns()
            .simple_impute()
            .rare_encode()
            .drop_saleprice_from_num()
            .cap_outliers()
            .drop_zero_variance()
            .finalize()
        )

