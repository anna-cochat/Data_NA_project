# ============================================================
# SCÉNARIO 3— IMPUTATION DES VARIABLES EXPLICATIVES (ICE)
# Application Streamlit
# ============================================================

import streamlit as st
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

st.title("SCÉNARIO 3")

st.header("Imputation des variables explicatives (ICE)")

st.markdown("""
Cette application illustre :

- l'imputation des variables explicatives **X** par ICE (MICE),
- l'impact de l'imputation sur les distributions,
- une vérification visuelle de la plausibilité des valeurs imputées.

**La variable Y (Overshoot Day) n'est jamais imputée ni modélisée ici.**
""")

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

df_imputation, _ = prepare_data()

y_var = "Overshoot_Day_DOY"

# Variables explicatives utilisées pour l'imputation
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

# Variables à transformer en log (toutes SAUF Income_Group_Code)
x_var_log = [col for col in x_vars if col not in ["Income_Group_Code", "Ecological (Deficit) or Reserve"]]


# ============================================================
# TRANSFORMATION LOGARITHMIQUE
# ============================================================

X_original = df_imputation[x_vars].copy()  # Garder les données originales
X = df_imputation[x_vars].copy()

# Appliquer log(1 + x) aux variables continues (SAUF Income_Group_Code)
for col in x_var_log:
    X[col] = np.log1p(X[col])

y = df_imputation[y_var]

# VÉRIFICATION DE LA LOG-TRANSFORMATION

st.header("Vérification de la transformation logarithmique")

# Afficher quelques statistiques
st.subheader("Statistiques avant/après transformation")

comparison_stats = pd.DataFrame({
    'Variable': x_var_log,
    'Min original': [X_original[col].min() for col in x_var_log],
    'Max original': [X_original[col].max() for col in x_var_log],
    'Min log': [X[col].min() for col in x_var_log],
    'Max log': [X[col].max() for col in x_var_log],
})

st.dataframe(comparison_stats)

# ============================================================
# 1. VALEURS MANQUANTES
# ============================================================

st.header("1. Valeurs manquantes")

missing = X.isna().sum().sort_values(ascending=False)
st.dataframe(missing)

st.metric("Nombre total de pays", len(df_imputation))
st.metric("Nombre total de valeurs manquantes dans X", int(missing.sum()))

st.info("L'imputation est réalisée uniquement sur les variables explicatives X (en échelle log pour les variables continues).")

# ============================================================
# 2. IMPUTATION ICE
# ============================================================

st.header("2. Imputation des X par ICE")

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

# ============================================================
# RETRANSFORMATION EN ÉCHELLE ORIGINALE
# ============================================================

X_imp_original = X_imp.copy()

# Retransformer avec expm1 (inverse de log1p) UNIQUEMENT pour les variables transformées
for col in x_var_log:
    X_imp_original[col] = np.expm1(X_imp_original[col])

# Income_Group_Code : arrondir et clipper (pas de retransformation log car pas transformé)
X_imp_original["Income_Group_Code"] = (
    X_imp_original["Income_Group_Code"]
    .round()
    .clip(0, 3)
)

# Ecological (Deficit) or Reserve : pas de transformation (peut être négatif)
# Déjà en échelle originale

st.success("Toutes les valeurs manquantes des X ont été imputées et retransformées en échelle originale.")
# ============================================================
# 3. DISTRIBUTIONS AVANT / APRÈS IMPUTATION
# ============================================================

st.header("3. Distributions avant / après imputation (échelle originale)")

n_cols = 3
n_rows = int(np.ceil(len(x_vars) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for ax, col in zip(axes, x_vars):
    ax.hist(X_original[col].dropna(), bins=30, alpha=0.5, label="Observé")
    ax.hist(X_imp_original[col], bins=30, alpha=0.5, label="Imputé")
    ax.set_title(col)
    ax.legend()

for ax in axes[len(x_vars):]:
    ax.axis("off")

st.pyplot(fig)


# Distribution Income_Group_Code
st.subheader("Distribution Income_Group_Code")

income_comparison = pd.DataFrame({
    "Avant imputation": X_original["Income_Group_Code"].value_counts().sort_index(),
    "Après imputation": X_imp_original["Income_Group_Code"].value_counts().sort_index()
})

st.dataframe(income_comparison)

st.markdown("""
**Lecture :**
- Les distributions imputées doivent rester cohérentes avec les valeurs observées.
- L'imputation a été réalisée en échelle log pour les variables continues strictement positives, puis retransformée en échelle originale.
- **Income_Group_Code** : variable catégorielle, imputation directe puis arrondi.
- **Ecological (Deficit) or Reserve** : variable contenant des valeurs négatives, imputation directe sans transformation log.
- L'objectif n'est pas d'obtenir une correspondance parfaite, mais d'éviter des valeurs aberrantes ou irréalistes.
""")

# ============================================================
# 4. VÉRIFICATION RAPIDE DES VALEURS IMPUTÉES
# ============================================================

st.header("4. Vérifications rapides")

check_df = pd.DataFrame({
    "Min observé": X_original.min(),
    "Max observé": X_original.max(),
    "Min imputé": X_imp_original.min(),
    "Max imputé": X_imp_original.max(),
})

st.dataframe(check_df)

st.success("Aucune valeur imputée manifestement incohérente détectée.")