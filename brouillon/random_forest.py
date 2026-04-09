import pandas as pd
import numpy as np
import statsmodels.api as sm

from nettoyage import prepare_data

from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib


# --------------------------------------------------
# DATA
# --------------------------------------------------

df_model = pd.read_csv("df_model.csv", index_col=0)
df_imputation = pd.read_csv("df_imputation.csv", index_col=0)


# --------------------------------------------------
# VARIABLES
# --------------------------------------------------

y_var = "Overshoot_Day_DOY"

terms_all = [
    'SDGi','Life Expectancy','HDI','Per Capita GDP','Population (millions)',
    'Cropland_Footprint_Production','Grazing_Footprint_Production','Forest_Footprint_Production',
    'Fish_Footprint_Production','BuiltUp_Footprint_Production','Carbon_Footprint_Production',
    'Cropland_Footprint_Consumption','Grazing_Footprint_Consumption','Forest_Footprint_Consumption',
    'Fish_Footprint_Consumption','Carbon_Footprint_Consumption',
    'Cropland','Grazing land','Forest land','Fishing ground',
    'Ecological (Deficit) or Reserve','Number of Earths required','Number of Countries required',
    'Income Group_LM','Income Group_UM','Income Group_HI',
    'Quality Score_2B','Quality Score_2C','Quality Score_3A',
    'Region_Asia-Pacific','Region_Central America/Caribbean','Region_EU',
    'Region_Middle East/Central Asia','Region_North America','Region_Other Europe','Region_South America'
]

terms_no_balance = [
    c for c in terms_all if c not in [
        'Ecological (Deficit) or Reserve',
        'Number of Earths required',
        'Number of Countries required'
    ]
]

# FIX 1 : on ne garde que les colonnes qui existent dans le DataFrame
terms_all        = [c for c in terms_all        if c in df_model.columns]
terms_no_balance = [c for c in terms_no_balance if c in df_model.columns]

# Colonnes manquantes (diagnostic)
missing = [c for c in terms_all if c not in df_model.columns]
if missing:
    print("⚠️  Colonnes absentes du CSV :", missing)


# --------------------------------------------------
# FIX 2 : dropna() AVANT le split, sur les colonnes utiles seulement
# → évite de perdre la moitié du dataset sur des colonnes hors modèle
# --------------------------------------------------

cols_needed = terms_all + [y_var]
df_clean = df_model.dropna(subset=cols_needed)

print(f"Lignes après dropna ciblé : {len(df_clean)}  (avant : {len(df_model)})")

train_df, test_df = train_test_split(
    df_clean,
    test_size=0.2,
    random_state=123
)

print("Train lignes:", len(train_df))
print("Test lignes: ", len(test_df))

# Plus besoin de dropna() sur train_df / test_df, tout est déjà propre
train_cc = train_df.copy()
test_cc  = test_df.copy()


# --------------------------------------------------
# STEPWISE OLS (backward AIC)
# --------------------------------------------------

def backward_stepwise_aic(X, y):

    X = sm.add_constant(X)
    remaining = list(X.columns)
    current_aic = sm.OLS(y, X[remaining]).fit().aic
    final_model = sm.OLS(y, X[remaining]).fit()

    while True:

        candidates = []

        for var in remaining:
            if var == "const":
                continue

            trial = [v for v in remaining if v != var]
            model = sm.OLS(y, X[trial]).fit()
            candidates.append((model.aic, var, model, trial))

        if not candidates:
            break

        best_aic, worst_var, best_model, best_vars = min(candidates, key=lambda x: x[0])

        if best_aic < current_aic:
            remaining    = best_vars
            current_aic  = best_aic
            final_model  = best_model
        else:
            break

    return final_model, remaining


# --------------------------------------------------
# CROSS VALIDATION (tree models)
# --------------------------------------------------

kf = KFold(n_splits=10, shuffle=True, random_state=123)

def evaluate_model(model, predictors):

    rmse_errors = []
    mae_errors  = []

    for train_idx, val_idx in kf.split(train_cc):

        fold_train = train_cc.iloc[train_idx]
        fold_val   = train_cc.iloc[val_idx]

        y_train = np.log(fold_train[y_var])
        y_val   = fold_val[y_var]

        # FIX 3 : on filtre sur les deux splits pour éviter les colonnes constantes
        valid_terms = [
            c for c in predictors
            if fold_train[c].nunique() > 1 and fold_val[c].nunique() >= 1
        ]

        X_train = fold_train[valid_terms]
        X_val   = fold_val[valid_terms]

        model.fit(X_train, y_train)

        pred_log = model.predict(X_val)
        pred     = np.exp(pred_log)

        rmse_errors.append(np.sqrt(mean_squared_error(y_val, pred)))
        mae_errors.append(mean_absolute_error(y_val, pred))

    return np.mean(rmse_errors), np.std(rmse_errors), np.mean(mae_errors)


# --------------------------------------------------
# OLS CROSS VALIDATION
# --------------------------------------------------

ols_errors = []
mae_errors_ols = []

for train_idx, val_idx in kf.split(train_cc):

    fold_train = train_cc.iloc[train_idx]
    fold_val   = train_cc.iloc[val_idx]

    y_train = np.log(fold_train[y_var])
    y_val   = fold_val[y_var]

    valid_terms = [c for c in terms_all if fold_train[c].nunique() > 1]

    X_train = fold_train[valid_terms]
    X_val   = fold_val[valid_terms]

    model_ols, selected_vars = backward_stepwise_aic(X_train, y_train)

    # selected_vars[0] == "const", on prend la suite
    pred_cols = [v for v in selected_vars if v != "const"]
    X_val_ols = sm.add_constant(X_val[pred_cols], has_constant='add')

    pred_log = model_ols.predict(X_val_ols)
    pred     = np.exp(pred_log)

    ols_errors.append(np.sqrt(mean_squared_error(y_val, pred)))
    mae_errors_ols.append(mean_absolute_error(y_val, pred))

ols_mean = np.mean(ols_errors)
ols_sd   = np.std(ols_errors)
ols_mae  = np.mean(mae_errors_ols)


# --------------------------------------------------
# TREE MODELS — CROSS VALIDATION
# --------------------------------------------------

models = {
    "RandomForest": RandomForestRegressor(n_estimators=500, random_state=123),
    "ExtraTrees":   ExtraTreesRegressor(n_estimators=500, random_state=123),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=3,
        random_state=123
    )
}

results = []

for name, model in models.items():
    mean_rmse, sd_rmse, mean_mae = evaluate_model(model, terms_all)
    results.append({
        "Model": name,
        "Predictors": "All variables",
        "RMSE": mean_rmse,
        "SD":   sd_rmse,
        "MAE":  mean_mae
    })

for name, model in models.items():
    mean_rmse, sd_rmse, mean_mae = evaluate_model(model, terms_no_balance)
    results.append({
        "Model": name,
        "Predictors": "No derived variables",
        "RMSE": mean_rmse,
        "SD":   sd_rmse,
        "MAE":  mean_mae
    })

results.append({
    "Model": "OLS stepwise (log)",
    "Predictors": "All variables",
    "RMSE": ols_mean,
    "SD":   ols_sd,
    "MAE":  ols_mae
})

df_results = pd.DataFrame(results).sort_values("RMSE")
print("\n=== CV RESULTS ===")
print(df_results.to_string(index=False))


# --------------------------------------------------
# MODÈLES FINAUX (entraînés sur tout le train)
# --------------------------------------------------

y_train_log = np.log(train_cc[y_var])

# --- Modèle 1 : Gradient Boosting (toutes variables) ---

valid_terms_all = [c for c in terms_all if train_cc[c].nunique() > 1]
X_train_all     = train_cc[valid_terms_all]

model_all = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    random_state=123
)
model_all.fit(X_train_all, y_train_log)

# --- Modèle 2 : Random Forest (sans variables dérivées) ---

valid_terms_clean = [c for c in terms_no_balance if train_cc[c].nunique() > 1]
X_train_clean     = train_cc[valid_terms_clean]

model_clean = RandomForestRegressor(n_estimators=500, random_state=123)
model_clean.fit(X_train_clean, y_train_log)


# --------------------------------------------------
# TEST SET EVALUATION
# --------------------------------------------------

y_test = test_cc[y_var]
print(f"\nTest lignes (après dropna ciblé) : {len(test_cc)}")

# Gradient Boosting
X_test_all = test_cc[valid_terms_all]
pred       = np.exp(model_all.predict(X_test_all))

rmse_all = np.sqrt(mean_squared_error(y_test, pred))
mae_all  = mean_absolute_error(y_test, pred)

print("\n=== GB TEST PERFORMANCE (toutes variables) ===")
print("RMSE:", round(rmse_all, 2))
print("MAE: ", round(mae_all,  2))

# Random Forest
X_test_clean = test_cc[valid_terms_clean]
pred         = np.exp(model_clean.predict(X_test_clean))

rmse_clean = np.sqrt(mean_squared_error(y_test, pred))
mae_clean  = mean_absolute_error(y_test, pred)

print("\n=== RF TEST PERFORMANCE (sans variables dérivées) ===")
print("RMSE:", round(rmse_clean, 2))
print("MAE: ", round(mae_clean,  2))