import pandas as pd
import yaml

from main import train_df, test_df
import src.utils.eda_helpers as hlp
from src.utils.eda_plt import plot_numeric_distributions, plot_categorical_features, plot_correlation_heatmap

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# hlp.data_overview(train_df)
# hlp.data_overview(test_df)


combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# def print_column_info(df):
#     for col in df.columns:
#         print(f"# {col} {df[col].dtype}")
#         unique_list = df[col].unique()
#         if len(unique_list) < 26:
#             print(f"# {unique_list}")
#
# print_column_info(combined_df)

cat_cols, num_cols, cat_but_car = hlp.grab_col_names(combined_df)


combined_df[num_cols].info()
combined_df[cat_cols].info()

hlp.missing_values_table(combined_df)

plot_numeric_distributions(combined_df, num_cols)

plot_categorical_features(combined_df,cat_cols)

for col in cat_cols:
    hlp.cat_summary(combined_df, col)

# Feature engineering
drop_candidates, dominance_table = hlp.dominant_class_filter(combined_df, cat_cols)
# drop_candidates  = ['Street', 'Utilities', 'LandSlope', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC', 'MiscFeature', 'KitchenAbvGr', 'PoolArea']


combined_df = combined_df.drop(columns=drop_candidates)



hlp.missing_values_table(combined_df)

combined_df, changed_cols = hlp.fill_structural_nans(combined_df)

combined_df["SalePrice"] = combined_df["SalePrice"].fillna(0)

hlp.missing_values_table(combined_df)

cat_cols, num_cols, cat_but_car = hlp.grab_col_names(combined_df)

# ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea',
# 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch', 'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']


combined_df[num_cols].info()
combined_df[cat_cols].info()
# Feature Engineering
numeric_in_cat= ["OverallQual", "OverallCond", "BsmtFullBath", "BsmtHalfBath",
                 "FullBath", "HalfBath", "BedroomAbvGr", "TotRmsAbvGrd",
                 "Fireplaces", "GarageCars", "MoSold", "YrSold"]



combined_df, cat_cols, num_cols = hlp.reclassify_columns(combined_df, cat_cols, num_cols, numeric_in_cat)

hlp.missing_values_table(combined_df)

combined_df = hlp.simple_imputer(combined_df, num_cols, cat_cols)

hlp.missing_values_table(combined_df)

##
plot_numeric_distributions(combined_df, num_cols)

plot_categorical_features(combined_df,cat_cols)

##
for col in cat_cols:
    hlp.cat_summary(combined_df, col)

hlp.rare_analyser(combined_df, "SalePrice",cat_cols)

combined_df = hlp.rare_encoder(combined_df,cat_cols, 0.1)

hlp.rare_analyser(combined_df, "SalePrice", cat_cols)


num_cols = [col for col in num_cols if col != "SalePrice"]


combined_df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T

for col in num_cols:
    print(col, hlp.check_outlier(combined_df, col))

for col in num_cols:
    hlp.replace_with_thresholds(combined_df, col)

for col in num_cols:
    print(col, hlp.check_outlier(combined_df, col))

plot_numeric_distributions(combined_df, num_cols)
plot_categorical_features(combined_df,cat_cols)

combined_df.describe([0, 0.01, 0.50, 0.95, 0.99, 1]).T

combined_df, num_cols, drop_cols = hlp.drop_zero_mean(combined_df, num_cols)
drop_cols


hlp.missing_values_table(combined_df)

for col in num_cols:
    print(col, hlp.check_outlier(combined_df, col))

########################################## Feature Extraction
import pandas as pd
import yaml
import pickle
import numpy as np


import src.utils.eda_helpers as hlp
from src.utils.eda_plt import plot_numeric_distributions, plot_categorical_features, plot_correlation_heatmap, get_strong_correlations
from src.data_loader import load_data

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

train_df, test_df = load_data(config)
train_df.shape
test_df.shape

combined_processed_path = config['data_paths']['combined_processed']
cat_path = config["data_paths"]["cat_cols"]
num_path = config["data_paths"]["num_cols"]


df = pd.read_csv(combined_processed_path)

with open(cat_path, "rb") as f:
    cat_cols = pickle.load(f)

with open(num_path, "rb") as f:
    num_cols = pickle.load(f)


df.info()

hlp.data_overview(df)

print("Kategorik sütunlar:", cat_cols)
print("Sayısal sütunlar:", num_cols)

strong_df = get_strong_correlations(df, cols=num_cols, threshold=0.3, output="list")

print(strong_df)


plot_correlation_heatmap(df, cols=num_cols)


plot_numeric_distributions(df, num_cols)

plot_categorical_features(df,cat_cols)

df.describe([0, 0.01, 0.50, 0.95, 0.99, 1]).T

# Toplam yaşam alanı
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
# Toplam banyo sayısı (yarım banyolar yarı olarak)
df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
# Oda başına alan
df["SFPerRoom"] = np.where(df["TotRmsAbvGrd"] != 0, df["TotalSF"] / df["TotRmsAbvGrd"], 0)

# Garaj başına araç alanı
df["AreaPerCar"] = np.where(df["GarageCars"] != 0, df["GarageArea"] / df["GarageCars"], 0)


# Ev yaşı, son tadilat yaşı ve garaj yaşı
df["HouseAge"]      = df["YrSold"] - df["YearBuilt"]
df["RemodelAge"]    = df["YrSold"] - df["YearRemodAdd"]
df["GarageAge"]     = df["YrSold"] - df["GarageYrBlt"]
# Bodrumun bitmiş alanı yüzdesi.
df["BsmtPctFinished"] = np.where( df["TotalBsmtSF"] != 0, df["BsmtFinSF1"] / df["TotalBsmtSF"], 0)

# toplam yaşam alanına göre kaplama alanı oranı
df["MasonryPct"] = df["MasVnrArea"] / df["TotalSF"]

new_num_feats = [
    "TotalSF",
    "TotalBath",
    "SFPerRoom",
    "AreaPerCar",
    "HouseAge",
    "RemodelAge",
    "GarageAge",
    "BsmtPctFinished",
    "MasonryPct"
]

num_cols.extend(new_num_feats)


plot_numeric_distributions(df, num_cols)

for col in num_cols:
    print(col, hlp.check_outlier(df, col))

df.info()

# Encoding
df = pd.get_dummies(df, columns=cat_cols, drop_first=True, prefix_sep="_", dtype=int)

train_df = df[df["SalePrice"] != 0].reset_index(drop=True)
test_df  = df[df["SalePrice"] == 0].reset_index(drop=True)

X_train = train_df.drop(columns="SalePrice")
y_train = train_df["SalePrice"]

X_test  = test_df.drop(columns="SalePrice")

print(f"Train örnek sayısı: {X_train.shape}, Test örnek sayısı: {X_test.shape}")


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
import joblib

# K-Fold tanımı
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Modellerin tanımı
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_jobs=-1, verbosity=0),
    "LightGBM": LGBMRegressor(random_state=42),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
}

# Sonuçları saklayacağımız liste
results = []

# Log dönüşümlü hedef
y_train_log = np.log1p(y_train)

for name, model in models.items():
    # 1. Doğrudan hedef ile RMSE
    neg_mse = cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
    rmse_direct = np.sqrt(-neg_mse).mean()

    # 2. Log dönüşümlü hedef ile CV tahmini al
    neg_mse_log = cross_val_score(model, X_train, y_train_log, cv=cv, scoring="neg_mean_squared_error")
    y_pred_log = cross_val_predict(model, X_train, y_train_log, cv=cv)
    y_pred_inv = np.expm1(y_pred_log)
    rmse_log_inv = np.sqrt(mean_squared_error(y_train, y_pred_inv))

    results.append((name, round(rmse_direct, 2), round(rmse_log_inv, 2)))

# Sonuçları tablo olarak göster
df_results = pd.DataFrame(results, columns=["Model", "RMSE (Doğrudan)", "RMSE (Log inverse)"])
print("\nRMSE Karşılaştırma Tablosu:\n")
print(df_results.sort_values(by="RMSE (Log inverse)"))


from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor

# Temel model
catboost_model = CatBoostRegressor(verbose=0, random_state=42)

# Parametre aralığı
param_dist = {
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'iterations': [250, 500, 750],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'border_count': [32, 64, 128]
}

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    catboost_model,
    param_distributions=param_dist,
    n_iter=25,
    scoring='neg_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# log1p dönüşümlü hedef ile eğitim
y_train_log = np.log1p(y_train)
random_search.fit(X_train, y_train_log)

# En iyi tahminleri al, inverse log dönüşüm uygula
y_pred_log = random_search.predict(X_train)
y_pred_inv = np.expm1(y_pred_log)
rmse_best = np.sqrt(mean_squared_error(y_train, y_pred_inv))

# Sonuçları yazdır
print("\n🔧 En iyi parametreler (RandomizedSearchCV):")
print(random_search.best_params_)
print(f"🎯 En iyi RMSE (inverse log space): {round(rmse_best, 2)}")


from sklearn.model_selection import GridSearchCV

# RandomizedSearch'ten gelen en iyi değerler
best_params = random_search.best_params_

# Grid oluşturuluyor — etrafındaki değerlerle
grid_params = {
    'depth': [best_params['depth'] - 1, best_params['depth'], best_params['depth'] + 1],
    'learning_rate': [best_params['learning_rate'] * 0.8, best_params['learning_rate'], best_params['learning_rate'] * 1.2],
    'l2_leaf_reg': [max(1, best_params['l2_leaf_reg'] - 1), best_params['l2_leaf_reg'], best_params['l2_leaf_reg'] + 1]
}

# GridSearchCV yapılandırması
grid_search = GridSearchCV(
    CatBoostRegressor(
        iterations=best_params['iterations'],
        border_count=best_params['border_count'],
        verbose=0,
        random_state=42
    ),
    param_grid=grid_params,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Eğitim ve tahmin
grid_search.fit(X_train, y_train_log)

# Inverse log ile RMSE
y_pred_log_grid = grid_search.predict(X_train)
y_pred_inv_grid = np.expm1(y_pred_log_grid)
rmse_grid = np.sqrt(mean_squared_error(y_train, y_pred_inv_grid))

print("\n🔎 GridSearchCV En İyi Parametreler:")
print(grid_search.best_params_)
print(f"🎯 GridSearchCV ile RMSE (inverse log space): {round(rmse_grid, 2)}")

test_ids_path = config["data_paths"]["test_ids"]

y_test_pred_log = grid_search.predict(X_test)
y_test_pred = np.expm1(y_test_pred_log)

id_df = pd.read_csv(test_ids_path)

submission_df = pd.DataFrame({
    "Id": id_df["Id"],
    "SalePrice": y_test_pred
})

submission_path = config["output_paths"]["submission"]
submission_df.to_csv(f"{submission_path}/catboost_gridsearch_submission.csv", index=False)
print("✅ Kaggle submission dosyası oluşturuldu: catboost_gridsearch_submission.csv")

joblib.dump(grid_search.best_estimator_, "models/final_catboost_model.pkl")

