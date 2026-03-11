import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from nettoyage import prepare_data

print("SCENARIO 2 : Prédiction des Y manquants (OLS & Random Forest)")

# ============================================================
# Chargement des modèles entraînés (scénario 1)
# ============================================================
with open("step_model.pkl", "rb") as f:
    result = pickle.load(f)

with open("rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

terms_all = [
    'SDGi', 'Life Expectancy', 'HDI', 'Per Capita GDP', 'Population (millions)',
    'Cropland_Footprint_Production', 'BuiltUp_Footprint_Production',
    'Cropland_Footprint_Consumption', 'Forest_Footprint_Consumption',
    'Fish_Footprint_Consumption', 'Cropland', 'Grazing land',
    'Ecological (Deficit) or Reserve', 'Number of Earths required',
    'Income Group_HI', 'Region_Other Europe'
]

# ============================================================
# Chargement des données
# ============================================================
df_imputation, df_model = prepare_data()
y_var = "Overshoot_Day_DOY"

# ============================================================
# Isolation des lignes avec Y manquant
# ============================================================
df_missing_y = df_model[df_model[y_var].isna()].copy()
print(f"Nombre de Y manquants à prédire : {len(df_missing_y)}")

if len(df_missing_y) == 0:
    print("Aucune valeur manquante à prédire.")
else:
    X_missing = df_missing_y[terms_all]

    # Prédiction OLS
    X_missing_const = sm.add_constant(X_missing, has_constant="add")
    y_pred_ols = result.predict(X_missing_const)

    # Prédiction Random Forest
    y_pred_rf = rf.predict(X_missing)

    # ============================================================
    # Tableau comparatif
    # ============================================================
    df_compare = pd.DataFrame({
        "OLS_DOY":    y_pred_ols.values,
        "RF_DOY":     y_pred_rf,
        "Diff_jours": np.abs(y_pred_ols.values - y_pred_rf)
    }, index=df_missing_y.index)

    df_compare["OLS_Date"] = pd.to_datetime("2018-01-01") + \
        pd.to_timedelta(df_compare["OLS_DOY"] - 1, unit="D")
    df_compare["RF_Date"] = pd.to_datetime("2018-01-01") + \
        pd.to_timedelta(df_compare["RF_DOY"] - 1, unit="D")

    df_compare = df_compare.sort_values("RF_Date")

    print("\nPrédictions des Y manquants :")
    print(df_compare[["OLS_Date", "RF_Date", "Diff_jours"]].to_string())

    # ============================================================
    # Graphique comparatif OLS vs RF
    # ============================================================
    plt.figure(figsize=(14, 6))
    plt.scatter(
        df_compare.index.astype(str),
        df_compare["OLS_Date"],
        alpha=0.7, label="OLS", marker="o"
    )
    plt.scatter(
        df_compare.index.astype(str),
        df_compare["RF_Date"],
        alpha=0.7, label="Random Forest", marker="^"
    )
    plt.xlabel("Pays imputés")
    plt.ylabel("Date prédite de l'Overshoot Day")
    plt.title("Scénario 2 – Y manquants : OLS vs Random Forest")

    ax = plt.gca()
    ax.yaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

    plt.xticks(rotation=90)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ============================================================
    # Sauvegarde du dataset complété (valeurs RF imputées)
    # ============================================================
    df_model_completed = df_model.copy()
    df_model_completed.loc[df_missing_y.index, y_var] = y_pred_rf
    df_model_completed.to_csv("data_completed.csv", index=False)
    print("\nDataset complété (RF) sauvegardé dans 'data_completed.csv'")

print("FIN SCENARIO 2")