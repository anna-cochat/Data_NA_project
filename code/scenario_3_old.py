# ============================================================
# SCENARIO 3 — IMPUTATION DES X ET COMPARAISON DES MODELES
# Application Streamlit pour la présentation
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

import statsmodels.api as sm

# ============================================================
# TITRE ET CONTEXTE
# ============================================================

st.title("Scénario 3 — Imputation des X et comparaison des modèles")

st.markdown("""
Cette application illustre :

- l'imputation des variables explicatives (ICE),
- l'impact de l'imputation sur les distributions des X,
- la comparaison *avant / après imputation* pour :
  - un modèle linéaire,
  - un random forest.

*La variable Y (Overshoot Day) n'est jamais imputée.*
""")

# ============================================================
# CHARGEMENT ET NETTOYAGE DES DONNÉES
# ============================================================

st.markdown("""
Étape 1 — Chargement et nettoyage des données

1. Charger la base de données pays depuis un fichier Excel
2. Renommer certaines variables pour assurer la cohérence des noms
3. Identifier les variables textuelles (pays, région, groupe de revenu, etc.)
4. Convertir toutes les autres variables en format numérique :
      - Remplacer les symboles non numériques
      - Transformer les valeurs invalides en valeurs manquantes
""")

@st.cache_data
def load_data():
    df = pd.read_excel("NA.xlsx")

    rename_dict = {
        "actual \nCountry Overshoot Day \n2018": "Overshoot Day",
        "Cropland Footprint.1": "Cropland_Footprint_Consumption",
        "Grazing Footprint.1": "Grazing_Footprint_Consumption",
        "Forest Product Footprint.1": "Forest_Footprint_Consumption",
        "Fish Footprint.1": "Fish_Footprint_Consumption",
        "Built up land.1": "BuiltUp_Footprint_Consumption",
    }

    df = df.rename(columns={c: rename_dict[c] for c in df.columns if c in rename_dict})

    force_text = ["Country", "Region", "Income Group", "Overshoot Day", "Quality Score"]

    def clean_numeric(s):
        if s.dtype != object:
            return s
        s = s.astype(str)
        s = s.replace(["-", "--", "", "…"], np.nan)
        s = s.str.replace(r"[,$% ]", "", regex=True)
        return pd.to_numeric(s, errors="coerce")

    for col in df.columns:
        if col not in force_text:
            df[col] = clean_numeric(df[col])

    return df


df = load_data()

st.markdown("""
Étape 2 — Construction de la variable cible Y

5. Transformer la date du “Overshoot Day” en jour de l'année (1-365)
6. Autoriser la présence de valeurs manquantes pour Y
      --> aucune suppression d'observations à ce stade
""")


# Transformation de Y en jour de l'année
df["Overshoot_Day_DOY"] = pd.to_datetime(
    df["Overshoot Day"], errors="coerce"
).dt.dayofyear

# Encodage ordinal du groupe de revenu
df["Income Group"] = pd.Categorical(
    df["Income Group"],
    categories=["LI", "LM", "UM", "HI"],
    ordered=True
)
df["Income_Group_Code"] = df["Income Group"].cat.codes

# Log du PIB
df["log_GDP_pc"] = np.log(df["Per Capita GDP"])

# Suppression des colonnes purement textuelles
df_model = df.drop(columns=[
    "Country", "Region", "Overshoot Day", "Quality Score", "Income Group"
])


# ============================================================
# DÉFINITION DES VARIABLES
# ============================================================


st.markdown("""
Étape 3 — Préparation des variables explicatives X

7. Définir le groupe de revenu comme variable ordinale (LI < LM < UM < HI)
8. Encoder ce groupe sous forme numérique pour l'imputation
9. Appliquer une transformation logarithmique au PIB par habitant
10. Supprimer uniquement les colonnes purement textuelles
""")

y_var = "Overshoot_Day_DOY"

x_vars = [
    "log_GDP_pc",
    "Cropland_Footprint_Consumption",
    "Grazing_Footprint_Consumption",
    "Forest_Footprint_Consumption",
    "Fish_Footprint_Consumption",
    "BuiltUp_Footprint_Consumption",
    "Grazing land",
    "Forest land",
    "Fishing ground",
    "Income_Group_Code",
]

st.markdown("""
Étape 4 — Définition des ensembles X et Y

11. Définir Y = Overshoot Day (jour de l'année)
12. Définir X comme l'ensemble des variables explicatives :
      - Niveau de richesse (log PIB)
      - Empreintes de consommation
      - Surfaces écologiques
      - Variable auxiliaire de revenu
""")

X = df_model[x_vars]
y = df_model[y_var]


# ============================================================
# 1. VALEURS MANQUANTES
# ============================================================

st.markdown("""
Étape 5 — Diagnostic de la structure des données

13. Compter le nombre total d'observations
14. Identifier :
      - Le nombre de Y observés
      - Le nombre de Y manquants
15. Vérifier la structure de la valeur manquante dans X et Y
""")

st.header("1. Données et valeurs manquantes")

missing = df_model[x_vars + [y_var]].isna().sum()
st.dataframe(missing)

st.metric("Nombre total de lignes", len(df_model))
st.metric("Y observé", y.notna().sum())
st.metric("Y manquant", y.isna().sum())

st.info(" L'imputation est réalisée uniquement sur les variables explicatives X.")


# ============================================================
# 2. IMPUTATION ICE DES X
# ============================================================

st.markdown("""
Étape 6 — Imputation multiple des variables X uniquement (ICE)

16. Appliquer une imputation itérative de type ICE sur X :
      - Chaque variable manquante est modélisée conditionnellement aux autres X
      - Le modèle conditionnel est une régression bayésienne
      - Le processus est itératif jusqu'à stabilisation
            
17. Obtenir une matrice X complète sans aucune valeur manquante
    Y n'est jamais utilisée pour l'imputation Y reste manquante si elle l'était initialement
""")

st.header("2. Imputation des variables explicatives (ICE)")

imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,              # 10 itérations ICE
    random_state=123,
    sample_posterior=True
)

X_imp = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

st.success("Toutes les valeurs manquantes des X ont été imputées.")


# ============================================================
# 3. DISTRIBUTIONS AVANT / APRÈS IMPUTATION
# ============================================================

st.markdown("""
Étape 7 — Validation visuelle de l'’'imputation

18. Pour chaque variable X :
      - Comparer la distribution observée initiale
      - À la distribution après imputation
19. Superposer les deux distributions sous forme d'histogrammes
20. Vérifier visuellement :
      - Cohérence des plages de valeurs
      - Absence de distorsion majeure
      - Respect de la structure des données
""")

st.header("3. Distributions des X avant et après imputation")

fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

for ax, col in zip(axes, X.columns):
    ax.hist(X[col], bins=30, alpha=0.5, label="Original")
    ax.hist(X_imp[col], bins=30, alpha=0.5, label="Imputé")
    ax.set_title(col)
    ax.legend()

for ax in axes[len(X.columns):]:
    ax.axis("off")

st.pyplot(fig)

st.markdown("""
*Lecture du graphique :*
- Bleu = valeurs observées (avec des manquants)
- Orange = valeurs après imputation ICE
- L'objectif est de vérifier que l'imputation *respecte la forme globale* des distributions.
""")


# ============================================================
# 4. COMPARAISON DES MODÈLES LINÉAIRES
# ============================================================

st.header("4. Comparaison des modèles linéaires")

# AVANT imputation : complete-case
df_cc = df_model[x_vars + [y_var]].dropna()
X_cc = sm.add_constant(df_cc[x_vars])
y_cc = df_cc[y_var]
lm_before = sm.OLS(y_cc, X_cc).fit()

# APRÈS imputation
df_complete = X_imp.copy()
df_complete[y_var] = y
df_train = df_complete[df_complete[y_var].notna()]

X_after = sm.add_constant(df_train[x_vars])
y_after = df_train[y_var]
lm_after = sm.OLS(y_after, X_after).fit()

fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

ax[0].scatter(y_cc, lm_before.fittedvalues, alpha=0.7)
ax[0].plot([y_cc.min(), y_cc.max()], [y_cc.min(), y_cc.max()], "r--")
ax[0].set_title("Modèle linéaire — avant imputation")
ax[0].set_xlabel("Observé")
ax[0].set_ylabel("Prédit")

ax[1].scatter(y_after, lm_after.fittedvalues, alpha=0.7)
ax[1].plot([y_after.min(), y_after.max()], [y_after.min(), y_after.max()], "r--")
ax[1].set_title("Modèle linéaire — après imputation")
ax[1].set_xlabel("Observé")

st.pyplot(fig)


# ============================================================
# 5. COMPARAISON DES RANDOM FORESTS
# ============================================================

st.header("5. Comparaison des Random Forests")

rf_before = RandomForestRegressor(n_estimators=500, random_state=123)
rf_before.fit(df_cc[x_vars], y_cc)

rf_after = RandomForestRegressor(n_estimators=500, random_state=123)
rf_after.fit(df_train[x_vars], y_after)

imp_df = pd.DataFrame({
    "Avant imputation": rf_before.feature_importances_,
    "Après imputation": rf_after.feature_importances_
}, index=x_vars)

fig, ax = plt.subplots(figsize=(8, 6))
imp_df.sort_values("Après imputation").plot(kind="barh", ax=ax)
ax.set_title("Random Forest — importance des variables")
ax.set_xlabel("Importance")

st.pyplot(fig)

# ======================================
# 4. PRÉDICTION DES Y MANQUANTS (LM)
# ======================================


st.header("4. Prédiction des Overshoot Day manquants")

st.markdown("""
À cette étape, on utilise un *modèle linéaire estimé après imputation des X*
pour *prédire les valeurs manquantes de la variable Y (Overshoot Day)*.

Rappel :
- seules les variables explicatives X sont imputées,
- Y n'est jamais imputée directement,
- les prédictions concernent uniquement les pays avec Y manquant.
""")

# --- Ré-estimation explicite du modèle linéaire ---
X_train_lm = sm.add_constant(
    df_train.drop(columns=[y_var]),
    has_constant="add"
)
y_train_lm = df_train[y_var]

lin_model = sm.OLS(y_train_lm, X_train_lm).fit()

# --- Sélection des Y manquants ---
df_y_missing = df_complete[df_complete[y_var].isna()].copy()

X_y_missing = sm.add_constant(
    df_y_missing.drop(columns=[y_var]),
    has_constant="add"
)

# --- Prédiction ---
df_y_missing["Overshoot_Day_Prédit"] = lin_model.predict(X_y_missing)

st.subheader("Aperçu des prédictions")
st.dataframe(
    df_y_missing[["Overshoot_Day_Prédit"]].round(1)
)

# --- Comparaison des distributions ---
st.subheader("Distribution : Y observé vs Y prédit")

fig, ax = plt.subplots(figsize=(7, 4))

ax.hist(
    df_train[y_var],
    bins=30,
    alpha=0.6,
    label="Y observé"
)

ax.hist(
    df_y_missing["Overshoot_Day_Prédit"],
    bins=30,
    alpha=0.6,
    label="Y prédit"
)

ax.set_xlabel("Overshoot Day (jour de l'année)")
ax.set_ylabel("Fréquence")
ax.legend()

st.pyplot(fig)

st.success(
    "Les Overshoot Day manquants ont été prédits à partir des X imputés "
    "via une régression linéaire."
)