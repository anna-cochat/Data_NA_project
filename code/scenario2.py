import pandas as pd
import numpy as np

from nettoyage import prepare_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# --------------------------------------------------
# CHARGEMENT DES DONNÉES
# --------------------------------------------------

# La fonction prepare_data() réalise le nettoyage initial
# et retourne deux jeux de données :
# df_imputation : version utilisée pour l'imputation
# df_model : version utilisée pour la modélisation

df_imputation, df_model = prepare_data()

# Variable cible à prédire
y_var = "Overshoot_Day_DOY"


# --------------------------------------------------
# LISTE DES VARIABLES EXPLICATIVES
# --------------------------------------------------

# Liste complète des variables explicatives disponibles
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

# Deuxième liste de variables :
# on retire les variables dérivées de l'équilibre écologique
# car elles sont très proches du calcul de l'Overshoot Day
terms_no_balance = [
    c for c in terms_all if c not in [
        'Ecological (Deficit) or Reserve',
        'Number of Earths required',
        'Number of Countries required'
    ]
]


# --------------------------------------------------
# MODÈLE 1 : GRADIENT BOOSTING (TOUTES LES VARIABLES)
# --------------------------------------------------

# On garde seulement les variables qui varient réellement
valid_terms_all = [c for c in terms_all if df_model[c].nunique() > 1]

# Lignes utilisables pour l'entraînement :
# Overshoot Day observé + toutes les variables explicatives disponibles
df_train_all = df_model[df_model[y_var].notna()].dropna(subset=valid_terms_all)

# Lignes à prédire :
# Overshoot Day manquant mais variables explicatives complètes
df_missing_all = df_model[df_model[y_var].isna()].dropna(subset=valid_terms_all)

print("Training rows (all variables):", len(df_train_all))
print("Rows to predict (all variables):", len(df_missing_all))

# Matrice des variables explicatives
X_train_all = df_train_all[valid_terms_all]

# On applique une transformation logarithmique à la variable cible
y_train_log = np.log(df_train_all[y_var])

# Modèle Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    random_state=123
)

# Entraînement du modèle
gb_model.fit(X_train_all, y_train_log)

# Variables explicatives pour les pays à prédire
X_pred_all = df_missing_all[valid_terms_all]

# Prédiction (sur l'échelle log)
pred_log = gb_model.predict(X_pred_all)

# Retour à l'échelle originale
pred_all = np.exp(pred_log)

# Stockage des prédictions
df_missing_all["Overshoot_pred_GB"] = pred_all


# --------------------------------------------------
# MODÈLE 2 : RANDOM FOREST (SANS VARIABLES DÉRIVÉES)
# --------------------------------------------------

# Même logique mais sans les variables dérivées
valid_terms_clean = [c for c in terms_no_balance if df_model[c].nunique() > 1]

df_train_clean = df_model[df_model[y_var].notna()].dropna(subset=valid_terms_clean)
df_missing_clean = df_model[df_model[y_var].isna()].dropna(subset=valid_terms_clean)

print("Training rows (no derived vars):", len(df_train_clean))
print("Rows to predict (no derived vars):", len(df_missing_clean))

# Matrice explicative
X_train_clean = df_train_clean[valid_terms_clean]

# Même transformation logarithmique
y_train_log = np.log(df_train_clean[y_var])

# Modèle Random Forest
rf_model = RandomForestRegressor(
    n_estimators=500,
    random_state=123
)

# Entraînement
rf_model.fit(X_train_clean, y_train_log)

# Prédiction pour les pays avec Overshoot manquant
X_pred_clean = df_missing_clean[valid_terms_clean]

pred_log = rf_model.predict(X_pred_clean)
pred_clean = np.exp(pred_log)

df_missing_clean["Overshoot_pred_RF_clean"] = pred_clean


# --------------------------------------------------
# FUSION DES RÉSULTATS
# --------------------------------------------------

# On crée une copie du jeu de données original
df_predictions = df_model.copy()

# Ajout des deux séries de prédictions
df_predictions.loc[df_missing_all.index, "Overshoot_pred_GB"] = df_missing_all["Overshoot_pred_GB"]
df_predictions.loc[df_missing_clean.index, "Overshoot_pred_RF_clean"] = df_missing_clean["Overshoot_pred_RF_clean"]


# --------------------------------------------------
# APERÇU DES PRÉDICTIONS
# --------------------------------------------------

print("\nPrediction preview\n")

print(df_predictions.loc[
    df_predictions[y_var].isna(),
    ["Overshoot_pred_GB","Overshoot_pred_RF_clean"]
].head())