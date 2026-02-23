# ============================================================
# SCÉNARIO — IMPUTATION DES VARIABLES EXPLICATIVES (ICE)
# Application Streamlit
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

from nettoyage import prepare_data

# ============================================================
# TITRE
# ============================================================

print("Imputation des variables explicatives (ICE)")

# Cette application illustre :

# - l’imputation des variables explicatives **X** par ICE (MICE),
# - l’impact de l’imputation sur les distributions,
# - une vérification visuelle de la plausibilité des valeurs imputées.

# La variable Y (Overshoot Day) n’est jamais imputée ni modélisée ici.

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

df_imputation, _ = prepare_data()

y_var = "Overshoot_Day_DOY"

# Variables explicatives utilisées pour l’imputation
x_vars = [
    "Income_Group_Code",
    "SDGi",
    "Life Expectancy",
    "HDI",
    "Per Capita GDP",
    "Population (millions)",
    "Cropland_Footprint_Production",
    "Grazing_Footprint_Production",
    "Forest_Footprint_Production",
    "Fish_Footprint_Production",
    "BuiltUp_Footprint_Production",
    "Carbon_Footprint_Production",
    "Cropland_Footprint_Consumption",
    "Grazing_Footprint_Consumption",
    "Forest_Footprint_Consumption",
    "Fish_Footprint_Consumption",
    "BuiltUp_Footprint_Consumption",
    "Carbon_Footprint_Consumption",
    "Cropland",
    "Grazing land",
    "Forest land",
    "Fishing ground",
    "BuiltUp_Biocapacity",
    "Total_Biocapacity",
    "Ecological (Deficit) or Reserve",
    "Number of Earths required",
    "Number of Countries required",
]

X = df_imputation[x_vars]
y = df_imputation[y_var]

# ============================================================
# 1. VALEURS MANQUANTES
# ============================================================

print("1. Valeurs manquantes")

missing = X.isna().sum().sort_values(ascending=False)
print(missing)

print("Nombre total de pays", len(df_imputation))
print("Nombre total de valeurs manquantes dans X", int(missing.sum()))

print("L'imputation est réalisée uniquement sur les variables explicatives X.")

# ============================================================
# 2. IMPUTATION ICE
# ============================================================

print("2. Imputation des X par ICE")

imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=123,
    sample_posterior=True
)

X_imp = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

X_imp["Income_Group_Code"] = (
    X_imp["Income_Group_Code"]
    .round()
    .clip(0, 3)
)

# ============================================================
# 3. DISTRIBUTIONS AVANT / APRÈS IMPUTATION
# ============================================================

print("3. Distributions avant / après imputation")

n_cols = 3
n_rows = int(np.ceil(len(x_vars) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for ax, col in zip(axes, x_vars):
    ax.hist(X[col], bins=30, alpha=0.5, label="Observé")
    ax.hist(X_imp[col], bins=30, alpha=0.5, label="Imputé")
    ax.set_title(col)
    ax.legend()

for ax in axes[len(x_vars):]:
    ax.axis("off")

print(fig)


# on represente l'income tout seul pas comme les autres par ce que en gros 
# limputation lis 0,1,2,3 comme valeurs possibles
# et impute de manière continue entre 0,1 1,2 et 2,3 puisqu'il comprend que 
# c'est ordonné du coup nous on arrondi au plus proche entier 
# et on clip pour que ca reste entre 0 et 3 et on visualise les distributions 
# avant et après imputation pour voir si ca a l'air cohérent ou pas
# jsp si cest la meilleure maniere de faire à voir avec les tutrices 

pd.DataFrame({
        "Avant imputation": X["Income_Group_Code"].value_counts(),
        "Après imputation": X_imp["Income_Group_Code"].value_counts()
})


# Lecture :
# - Les distributions imputées doivent rester cohérentes avec les valeurs observées.
# - L’objectif n’est pas d’obtenir une correspondance parfaite, mais d’éviter 
# des valeurs aberrantes ou irréalistes.

# ============================================================
# 4. VÉRIFICATION RAPIDE DES VALEURS IMPUTÉES
# ============================================================

print("4. Vérifications rapides")

check_df = pd.DataFrame({
    "Min observé": X.min(),
    "Max observé": X.max(),
    "Min imputé": X_imp.min(),
    "Max imputé": X_imp.max(),
})

print(check_df)

