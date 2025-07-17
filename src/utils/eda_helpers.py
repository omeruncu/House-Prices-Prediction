import os
from typing import Tuple, List

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def data_overview(df: pd.DataFrame) -> None:
    """
    Veri setinin genel yapısını özetler.

    Parametre:
    df (pd.DataFrame): Analiz edilecek veri seti

    Çıktı:
    - Gözlem ve özellik sayısı
    - Sütun isimleri
    - Veri tipleri ve bellek kullanımı
    - Temel istatistikler (özelleştirilmiş yüzdeliklerle)
    - Eksik değer özet tablosu
    """
    print("\n🧾 Veri Seti Genel Bilgisi")
    print(f"Gözlem sayısı : {df.shape[0]}")
    print(f"Özellik sayısı: {df.shape[1]}")
    print(f"Sütunlar      : {list(df.columns)}")

    print("\n📊 Veri Tipleri ve Bellek Kullanımı")
    df.info()

    print("\n📈 Temel İstatistikler (özelleştirilmiş yüzdeliklerle)")
    desc = df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T
    print(desc)

    print("\n🚨 Eksik (NaN) Değerler")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print(missing)
    else:
        print("Eksik değer bulunmuyor.")


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """
    DataFrame'i 'data/processed/' klasörüne belirtilen adla kaydeder.

    Parametre:
    df (pd.DataFrame): Kaydedilecek veri
    filename (str): Kaydedilecek dosya adı (örn. 'eda_output.csv')
    """
    dir_path = "data/processed"
    os.makedirs(dir_path, exist_ok=True)  # klasör yoksa oluştur
    full_path = os.path.join(dir_path, filename)
    df.to_csv(full_path, index=False)
    print(f"✅ Veri başarıyla kaydedildi: {full_path}")

def grab_col_names(df, cat_th=20, car_th=25, numeric_threshold=0.95):
    """
    Değişkenleri veri tipine ve eşiklere göre sınıflandırır.
    Object tipinde olup büyük oranda sayısal olan sütunları da dönüştürür.

    Parameters
    ----------
    df : pd.DataFrame
    cat_th : int
        Kategorik sayılabilecek değişkenler için sınıf sayısı eşiği
    car_th : int
        Kardinal değişkenler için eşik
    numeric_threshold : float
        Object sütunların sayıya çevrilebilirlik oranı eşiği

    Returns
    -------
    cat_cols : list
        Kategorik değişkenler
    num_cols : list
        Sayısal değişkenler
    cat_but_car : list
        Kategorik görünümlü ama kardinal olan değişkenler
    """

    object_but_numeric = []

    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = df[col].str.strip()
            converted = pd.to_numeric(cleaned, errors="coerce")
            ratio_numeric = converted.notna().mean()

            if ratio_numeric >= numeric_threshold:
                df[col] = converted
                object_but_numeric.append(col)

    # Güncellenmiş dtype'lara göre sınıflandır
    cat_cols = [col for col in df.columns if df[col].dtype in ["object", "category", "bool"]]
    cat_cols = [col for col in cat_cols if col not in object_but_numeric]

    num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and
                   df[col].dtype in ["int", "float"]]


    cat_but_car = [col for col in cat_cols if df[col].nunique() > car_th]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Toplam değişken sayısı: {df.shape[1]}")
    print(f"Kategorik değişken sayısı: {len(cat_cols)}")
    print(f"Sayısal değişken sayısı: {len(num_cols)}")
    print(f"Kategorik görünümlü ama kardinal değişken sayısı: {len(cat_but_car)}")

    return cat_cols, num_cols, cat_but_car

def summarize_column_types(df, cat_cols, num_cols, cat_but_car):
    """
    DataFrame'deki değişkenlerin türlerini özetleyen bir tablo üretir.

    Parameters:
    - df: pd.DataFrame
    - cat_cols: Kategorik değişkenler
    - num_cols: Sayısal değişkenler
    - cat_but_car: Kardinal kategorikler

    Çıktı:
    - Sütun türü özetleri
    """
    binary_cols = [col for col in cat_cols if df[col].nunique() == 2 and col not in ["Street", "Alley"]]
    nominal_cols = [col for col in cat_cols if df[col].nunique() > 2 and col not in cat_but_car]
    ordinal_candidates = [col for col in cat_cols if col in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond",
                                                             "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
                                                             "HeatingQC", "KitchenQual", "FireplaceQu", "GarageFinish",
                                                             "GarageQual", "GarageCond", "PoolQC", "Fence"]]

    return num_cols, nominal_cols, binary_cols, ordinal_candidates, cat_but_car

def dominant_class_filter(df, cat_cols, threshold=0.95):
    """
    Baskın sınıf oranı threshold'u aşan kategorik değişkenleri (NaN dahil) tespit eder.

    Parameters:
    - df: pd.DataFrame
    - cat_cols: Kategorik değişken listesi
    - threshold: Üst sınır (varsayılan: 0.95)

    Returns:
    - drop_candidates: Droplanabilecek sütun listesi
    - summary_df: Baskın oranları içeren tablo
    """
    drop_candidates = []
    summary = []

    for col in cat_cols:
        total = len(df)
        freq = df[col].fillna("Missing").value_counts()
        top_class_count = freq.iloc[0]
        top_class_ratio = top_class_count / total

        summary.append({
            "column": col,
            "top_class": freq.index[0],
            "top_class_count": top_class_count,
            "top_class_ratio": round(top_class_ratio, 4)
        })

        if top_class_ratio >= threshold:
            drop_candidates.append(col)

    summary_df = pd.DataFrame(summary).sort_values(by="top_class_ratio", ascending=False)

    print(f"{len(drop_candidates)} değişken önerilen threshold ({threshold}) üzerinde baskın sınıfa sahip.")
    return drop_candidates, summary_df


def fill_structural_nans(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Yapısal eksikliğe sahip değişkenlerde NaN değerleri anlamsal etiketle doldurur.

    Parameters:
    - df: pd.DataFrame

    Returns:
    - df: Doldurulmuş veri çerçevesi
    """

    replacements = {
        "Alley": "NA",
        "Fence": "NA",
        "MasVnrType": "None",
        "FireplaceQu": "NA",
        "GarageFinish": "NA",
        "GarageQual": "NA",
        "GarageCond": "NA",
        "GarageType": "NA",
        "BsmtExposure": "NA",
        "BsmtCond": "NA",
        "BsmtQual": "NA",
        "BsmtFinType1": "NA",
        "BsmtFinType2": "NA"
    }

    changed_cols = []
    for col, fill_val in replacements.items():
        if col in df.columns:
            df[col].fillna(fill_val, inplace=True)
            changed_cols.append(col)

    return df, changed_cols

def reclassify_columns(df, cat_cols, num_cols, numeric_in_cat):
    """
    cat_cols listesinden numeric_in_cat içindeki kolonları çıkartır,
    num_cols listesine ekler ve kalan cat_cols'u df üzerinde object tipine çevirir.

    Args:
        df (pd.DataFrame): Üzerinde dönüşüm yapılacak DataFrame.
        cat_cols (list): Başlangıçta kategorik kabul edilen kolon isimleri.
        num_cols (list): Başlangıçta numerik kabul edilen kolon isimleri.
        numeric_in_cat (list): cat_cols içinde olmasına rağmen numerik sayılması gereken kolonlar.

    Returns:
        updated_cat_cols (list): numeric_in_cat çıkarıldıktan sonraki kategorik kolon listesi.
        updated_num_cols (list): numeric_in_cat eklendikten sonraki numerik kolon listesi.
    """
    # 1. num_cols'u güncelle
    updated_num_cols = num_cols.copy()
    for col in numeric_in_cat:
        if col in cat_cols and col not in updated_num_cols:
            updated_num_cols.append(col)

    # 2. cat_cols'u güncelle
    updated_cat_cols = [col for col in cat_cols if col not in numeric_in_cat]

    # 3. kalan cat_cols'u df üzerinde object tipine çevir
    for col in updated_cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    return df, updated_cat_cols, updated_num_cols


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        vc = dataframe[col].value_counts()
        vr = vc / len(dataframe)
        tm = dataframe.groupby(col)[target].mean()

        result = pd.concat([vc, vr, tm], axis=1, sort=False)
        result.columns = ["COUNT", "RATIO", "TARGET_MEAN"]

        print(result, end="\n\n\n")


def rare_encoder(dataframe, cat_cols, rare_perc=0.01):
    """
    Belirtilen kategorik değişkenlerde nadir sınıfları 'Rare' etiketiyle birleştirir.

    Parameters:
    - dataframe: pd.DataFrame → Veri seti
    - cat_cols: list → Kategorik değişken listesi
    - rare_perc: float → Rare sınıf eşiği (varsayılan: %1)

    Returns:
    - encoded_df: Nadir etiketler dönüştürülmüş veri çerçevesi
    """
    temp_df = dataframe.copy()
    rare_columns = []

    for col in cat_cols:
        freqs = temp_df[col].value_counts(normalize=True, dropna=False)
        rare_labels = freqs[freqs < rare_perc].index

        if len(rare_labels) > 0:
            rare_columns.append(col)
            temp_df[col] = np.where(temp_df[col].isin(rare_labels), 'Rare', temp_df[col])

    print(f"Rare label encoding uygulanan değişkenler: {rare_columns}")
    return temp_df


def simple_imputer(df: pd.DataFrame, num_cols: list, cat_cols: list) -> pd.DataFrame:
    """
    Sayısal sütunlarda eksik değerleri medyan ile,
    kategorik sütunlarda eksik değerleri mode ile doldurur.

    Parameters:
    - df: pd.DataFrame → Veri seti
    - num_cols: list → Sayısal sütunlar
    - cat_cols: list → Kategorik sütunlar

    Returns:
    - df: Eksiklikleri doldurulmuş veri çerçevesi
    """

    df = df.copy()

    # Sayısal sütunlar → medyan ile doldur
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        if col == "GarageYrBlt":
            median_val = df[col].median()
            df.loc[df[col] > 2010, col] = median_val

    # Kategorik sütunlar → mode ile doldur
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode(dropna=True)[0]
            df[col] = df[col].fillna(mode_val)

    return df

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def drop_zero_mean(df, num_cols):
    """
    num_cols listesindeki, mean == 0 ve std == 0 olan sütunları
    DataFrame'den ve num_cols listesinden drop eder.

    Args:
        df (pd.DataFrame): İncelenecek veri çerçevesi.
        num_cols (list): Sayısal sütun isimlerini içeren liste.

    Returns:
        df (pd.DataFrame): Belirtilen sütunlar düşürülmüş yeni df.
        updated_num_cols (list): Kalan sayısal sütun isimleri.
        drop_cols (list): Düşürülen sütun isimleri.
    """
    # Drop edilecek sütunları belirle
    drop_cols = [
        col for col in num_cols
        if df[col].mean() == 0
    ]

    # DataFrame'den ve num_cols listesinden çıkar
    df = df.drop(columns=drop_cols)
    updated_num_cols = [col for col in num_cols if col not in drop_cols]

    return df, updated_num_cols, drop_cols

