import pickle
import pandas as pd
import statsmodels.api as sm
from nettoyage import prepare_data

print("===== SCENARIO 2 : Imputation des Y manquants =====")

# ==============================
# Chargement du modèle entraîné (scénario 1)
# ==============================
with open("step_model.pkl", "rb") as f:
    step_model = pickle.load(f)

with open("selected_vars.pkl", "rb") as f:
    selected_vars = pickle.load(f)

# ==============================
# Chargement des données
# ==============================
df_imputation, df_model = prepare_data()

y_var = "Overshoot_Day_DOY"

# Variables utilisées par le modèle (sans la constante)
vars_no_const = [v for v in selected_vars if v != "const"]

# ==============================
# Séparation données connues / manquantes
# ==============================

# Lignes où Y est manquant
df_missing_y = df_model[df_model[y_var].isna()].copy()

print(f"Nombre de Y manquants à prédire : {len(df_missing_y)}")

if len(df_missing_y) == 0:
    print("Aucune valeur manquante à prédire.")
else:
    # ==============================
    # Construction de X pour prédiction
    # ==============================
    X_missing = df_missing_y[vars_no_const]

    # Supprimer lignes avec NA dans X (impossible à prédire)
    mask_valid = X_missing.notna().all(axis=1)
    X_missing_clean = X_missing[mask_valid]

    print(f"Lignes exploitables pour prédiction : {len(X_missing_clean)}")

    # Ajouter constante
    X_missing_clean = sm.add_constant(X_missing_clean, has_constant="add")

    # ==============================
    # Prédiction des Y manquants
    # ==============================
    y_pred_missing = step_model.predict(X_missing_clean)

    # ==============================
    # Remplacement dans le dataset
    # ==============================
    df_model.loc[X_missing_clean.index, y_var] = y_pred_missing

    print("Imputation terminée.")

# ==============================
# 7️⃣ Sauvegarde du dataset complété
# ==============================
df_model.to_csv("data_completed.csv", index=False)

print("Dataset complété sauvegardé dans 'data_completed.csv'")
print("===== FIN SCENARIO 2 =====")