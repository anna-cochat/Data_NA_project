import pickle
import pandas as pd
import statsmodels.api as sm
from nettoyage import prepare_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

print(" SCENARIO 2 : Imputation des Y manquants")

# Chargement du modèle entraîné (scénario 1)
with open("step_model.pkl", "rb") as f:
    step_model = pickle.load(f)
with open("selected_vars.pkl", "rb") as f:
    selected_vars = pickle.load(f)

# données préparées pour le modèle (imputation déjà faite pour les X)
df_imputation, df_model = prepare_data()
y_var = "Overshoot_Day_DOY"

# Variables utilisées par le modèle (sans la constante)
vars_no_const = [v for v in selected_vars if v != "const"]

# Séparation données connues / manquantes

# Lignes où Y est manquant
df_missing_y = df_model[df_model[y_var].isna()].copy()

print(f"Nombre de Y manquants à prédire : {len(df_missing_y)}")

if len(df_missing_y) == 0:
    print("Aucune valeur manquante à prédire.")
else:
    # Construction de X pour prédiction
    X_missing = df_missing_y[vars_no_const]

    print(f"Lignes exploitables pour prédiction : {len(X_missing)}")

    # Ajouter constante
    X_missing_clean = sm.add_constant(X_missing, has_constant="add")

    # ==============================
    # Prédiction des Y manquants
    # ==============================
    y_pred_missing = step_model.predict(X_missing_clean)

    # ==============================
    # Remplacement dans le dataset
    # ==============================
    df_model.loc[X_missing_clean.index, y_var] = y_pred_missing

    print("Imputation terminée.")

# Sauvegarde du dataset complété
df_model.to_csv("data_completed.csv", index=False)

print("Dataset complété sauvegardé dans 'data_completed.csv'")
print("FIN SCENARIO 2")



# ==============================
# Récupérer UNIQUEMENT les lignes imputées
# ==============================

df_predicted = df_model.loc[X_missing_clean.index].copy()

# ==============================
# Conversion DOY -> Date (année fixe 2018 pour éviter erreur Year)
# ==============================

df_predicted["Overshoot_Date"] = pd.to_datetime("2018-01-01") + \
    pd.to_timedelta(df_predicted[y_var] - 1, unit="D")

# Trier par date
df_predicted = df_predicted.sort_values("Overshoot_Date")

# ==============================
# Graphique
# ==============================

plt.figure(figsize=(14, 6))

plt.scatter(
    df_predicted.index.astype(str),   # on utilise l’index si Country pose problème
    df_predicted["Overshoot_Date"]
)

plt.xlabel("Pays imputés")
plt.ylabel("Date prédite de l'Overshoot Day (2018)")
plt.title("Overshoot Day prédit pour les Y manquants")

ax = plt.gca()
ax.yaxis.set_major_locator(mdates.MonthLocator())
ax.yaxis.set_major_formatter(mdates.DateFormatter('%d %b'))

plt.xticks(rotation=90)
plt.grid(True)

plt.tight_layout()
plt.show()