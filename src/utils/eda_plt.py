import os
os.environ['MPLBACKEND'] = 'TkAgg'
import matplotlib
matplotlib.use('TkAgg')

import pandas as pd

from  matplotlib import pyplot as plt
import seaborn as sns


def plot_numeric_distributions(
    df,
    num_cols=None,
    bins=30,
    kde=True,
    figsize=(12, 4)
):
    """
    Her sayısal sütun için yanyana Histogram ve Boxplot çizer.

    Args:
      df        : pandas DataFrame
      num_cols  : list of column names; None ise tüm sayısal sütunlar seçilir
      bins      : histogram için bin sayısı
      kde       : histogram üstüne KDE eğrisi eklesin mi?
      figsize   : her grafiğin figsize parametresi
    """
    # Eğer num_cols verilmemişse, tüm sayısal sütunları al
    if num_cols is None:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Histogram + KDE
        sns.histplot(
            data=df,
            x=col,
            bins=bins,
            kde=kde,
            ax=axes[0]
        )
        axes[0].set_title(f"{col} – Histogram")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Frekans")

        # Boxplot
        sns.boxplot(
            x=df[col],
            ax=axes[1]
        )
        axes[1].set_title(f"{col} – Boxplot")
        axes[1].set_xlabel(col)

        plt.tight_layout()
        plt.show()


def plot_categorical_features(df, cat_cols, max_categories=25):
    """
    Kategorik sütunlardaki sınıf dağılımlarını çubuk grafiklerle görselleştirir.

    Parameters:
    - df: pd.DataFrame → Veri seti
    - cat_cols: list → Kategorik sütunlar
    - max_categories: int → Maksimum sınıf sayısı. Fazlası varsa grafik çizilmez.
    """

    for col in cat_cols:
        n_unique = df[col].nunique(dropna=False)

        if n_unique > max_categories:
            print(f"'{col}' değişkeni {n_unique} sınıfa sahip. Atlandı.")
            continue

        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, palette="Set3")
        plt.title(f"{col} Dağılımı ({n_unique} Sınıf)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df, cols=None, figsize=(16, 12), annot=True, cmap="coolwarm"):
    """
    Sayısal değişkenler için korelasyon ısı haritası oluşturur.

    Parameters:
        df (pd.DataFrame): Veri seti.
        cols (list): Korelasyon için kullanılacak sütunlar (varsayılan: tüm sayısal sütunlar).
        figsize (tuple): Görsel boyutu.
        annot (bool): Değer anotasyonları gösterilsin mi.
        cmap (str): Renk paleti.

    Returns:
        None: Gösterim yapılır.
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns

    corr_matrix = df[cols].corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f", linewidths=0.5)
    plt.title("Korelasyon Matrisi")
    plt.show()


def get_strong_correlations(df, cols=None, threshold=0.3, sort=True, output="list"):
    """
    Korelasyonu belirtilen eşik değerden yüksek olan çiftleri çıkarır.

    Parameters:
        df (pd.DataFrame): Veri seti
        cols (list): Sayısal sütunlar
        threshold (float): Korelasyon eşiği
        sort (bool): Büyükten küçüğe sırala
        output (str): "list" | "df"

    Returns:
        list or pd.DataFrame: Korelasyon çiftleri
    """
    if cols is None:
        cols = df.select_dtypes(include="number").columns

    corr_matrix = df[cols].corr()
    pairs = []

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            corr = corr_matrix.loc[c1, c2]

            if abs(corr) >= threshold:
                pairs.append((c1, c2, round(corr, 2)))

    if sort:
        pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)

    if output == "df":
        return pd.DataFrame(pairs, columns=["Feature 1", "Feature 2", "Correlation"])
    return pairs

