import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date

from nettoyage import prepare_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics.pairwise import euclidean_distances


# --------------------------------------------------
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# --------------------------------------------------

df_imputation, df_model = prepare_data()

variable_cible = "Overshoot_Day_DOY"


# --------------------------------------------------
# 2. LISTE DES VARIABLES EXPLICATIVES
# --------------------------------------------------

variables_exp_completes = [
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

variables_exp_sans_bilan = [
    v for v in variables_exp_completes if v not in [
        'Ecological (Deficit) or Reserve',
        'Number of Earths required',
        'Number of Countries required'
    ]
]


# --------------------------------------------------
# 3. DATASET UTILISABLE POUR SCENARIO 2
# (complete predictors)
# --------------------------------------------------

df_scenario2 = df_model.dropna(subset=variables_exp_completes)

print("Countries usable for Scenario 2:", len(df_scenario2))


# --------------------------------------------------
# 4. MODÈLE 1 : GRADIENT BOOSTING
# --------------------------------------------------

variables_valides_all = [
    c for c in variables_exp_completes if df_scenario2[c].nunique() > 1
]

df_train_all = df_scenario2[df_scenario2[variable_cible].notna()]
df_a_predire_all = df_scenario2[df_scenario2[variable_cible].isna()]

print("Lignes d'entraînement (toutes variables) :", len(df_train_all))
print("Pays à prédire :", len(df_a_predire_all))

X_train_all = df_train_all[variables_valides_all]
y_train_log = np.log(df_train_all[variable_cible])

modele_gb = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    random_state=123
)

modele_gb.fit(X_train_all, y_train_log)

X_pred_all = df_a_predire_all[variables_valides_all]

pred_log = modele_gb.predict(X_pred_all)
pred_gb = np.exp(pred_log)

df_a_predire_all["Overshoot_pred_GB"] = pred_gb


# --------------------------------------------------
# 5. MODÈLE 2 : RANDOM FOREST
# --------------------------------------------------

variables_valides_clean = [
    c for c in variables_exp_sans_bilan if df_scenario2[c].nunique() > 1
]

df_train_clean = df_scenario2[df_scenario2[variable_cible].notna()]
df_a_predire_clean = df_scenario2[df_scenario2[variable_cible].isna()]

print("Lignes d'entraînement (sans variables dérivées) :", len(df_train_clean))
print("Pays à prédire :", len(df_a_predire_clean))

X_train_clean = df_train_clean[variables_valides_clean]
y_train_log = np.log(df_train_clean[variable_cible])

modele_rf = RandomForestRegressor(
    n_estimators=500,
    random_state=123
)

modele_rf.fit(X_train_clean, y_train_log)

X_pred_clean = df_a_predire_clean[variables_valides_clean]

pred_log = modele_rf.predict(X_pred_clean)
pred_rf = np.exp(pred_log)

df_a_predire_clean["Overshoot_pred_RF"] = pred_rf


# --------------------------------------------------
# 6. FUSION DES PRÉDICTIONS
# --------------------------------------------------

df_predictions = df_scenario2.copy()

df_predictions.loc[df_a_predire_all.index, "Overshoot_pred_GB"] = df_a_predire_all["Overshoot_pred_GB"]
df_predictions.loc[df_a_predire_clean.index, "Overshoot_pred_RF"] = df_a_predire_clean["Overshoot_pred_RF"]

print("\nAperçu des prédictions\n")

print(
    df_predictions.loc[
        df_predictions[variable_cible].isna(),
        ["Overshoot_pred_GB","Overshoot_pred_RF"]
    ].head()
)


# --------------------------------------------------
# 7. CHARGEMENT DES COORDONNÉES FAMD
# --------------------------------------------------

coords = pd.read_csv("/Users/admin/Documents/GitHub/data_na_project/famd_coordinates.csv").set_index("Country")

print("Countries in FAMD coords:", len(coords.index))
print("Countries in scenario2 dataset:", len(df_scenario2.index))

eig = pd.read_csv("/Users/admin/Documents/GitHub/data_na_project/famd_eigenvalues.csv")

cumvar = eig["percentage of variance"].cumsum()
k = (cumvar >= 80).idxmax() + 1

print("\nDimensions FAMD utilisées :", k)

coords_k = coords.iloc[:, :k]


# --------------------------------------------------
# 8. MATRICE DES DISTANCES
# --------------------------------------------------

distances_famd = pd.DataFrame(
    euclidean_distances(coords_k),
    index=coords_k.index,
    columns=coords_k.index
)

print("\nAperçu matrice des distances\n")
print(distances_famd.iloc[:5, :5])


# --------------------------------------------------
# 9. VALIDATION PAR VOISINS
# --------------------------------------------------

pays_predits = df_predictions.loc[
    df_predictions[variable_cible].isna()
].index

lignes = []

for pays in pays_predits:

    if pays not in distances_famd.index:
        continue

    pred_gb = df_predictions.loc[pays, "Overshoot_pred_GB"]
    pred_rf = df_predictions.loc[pays, "Overshoot_pred_RF"]

    voisins = distances_famd.loc[pays].sort_values().index[1:6]

    for v in voisins:

        overshoot_obs = df_predictions.loc[v, variable_cible]

        lignes.append({

            "Pays_prédit": pays,
            "Prediction_GB": pred_gb,
            "Prediction_RF": pred_rf,
            "Voisin": v,
            "Distance_FAMD": distances_famd.loc[pays, v],
            "Statut_voisin":
                "Observé" if pd.notna(overshoot_obs) else "Manquant",
            "Overshoot_observé_voisin": overshoot_obs,
            "Prediction_voisin_GB":
                df_predictions.loc[v, "Overshoot_pred_GB"],
            "Prediction_voisin_RF":
                df_predictions.loc[v, "Overshoot_pred_RF"]
        })


table_voisins = pd.DataFrame(lignes)

table_voisins = table_voisins.sort_values(
    ["Pays_prédit","Distance_FAMD"]
)


# --------------------------------------------------
# 10. AFFICHAGE
# --------------------------------------------------

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

print("\nTable de validation par voisins\n")
print(table_voisins.head(20))

print("\nRésumé statut des voisins\n")
print(table_voisins["Statut_voisin"].value_counts())


# --------------------------------------------------
# 11. EXPORT
# --------------------------------------------------

#table_voisins.to_csv(
 #   "validation_voisins_famd_overshoot.csv",
  #  index=False
#)

#print("\nTable exportée : validation_voisins_famd_overshoot.csv")

df_predicted = df_predictions.loc[
    df_predictions[variable_cible].isna(),
    ["Overshoot_pred_GB", "Overshoot_pred_RF"]
].copy()

df_predicted = df_predicted.rename(columns={
    "Overshoot_pred_GB": "DOY_GB",
    "Overshoot_pred_RF": "DOY_RF"
})
df_predicted["Date_GB"] = df_predicted["DOY_GB"].apply(
    lambda d: date.fromordinal(date(2018,1,1).toordinal() + int(d) - 1)
)

df_predicted["Date_RF"] = df_predicted["DOY_RF"].apply(
    lambda d: date.fromordinal(date(2018,1,1).toordinal() + int(d) - 1)
)

df_predicted = df_predicted.sort_values("DOY_GB")

fig, ax = plt.subplots(figsize=(18,7))

x = np.arange(len(df_predicted))

ax.scatter(
    x,
    df_predicted["Date_GB"],
    color="#1f77b4",
    label="Gradient Boosting",
    s=50
)

ax.scatter(
    x,
    df_predicted["Date_RF"],
    color="#ff7f0e",
    label="Random Forest",
    s=50
)

ax.set_xticks(x)
ax.set_xticklabels(
    df_predicted.index,
    rotation=45,
    ha="right",
    fontsize=8
)

ax.set_xlabel("Pays imputés")
ax.set_ylabel("Date prédite de l'Overshoot Day (2018)")
ax.set_title("Comparaison des prédictions : Gradient Boosting vs Random Forest")

ax.yaxis_date()
ax.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.yaxis.set_major_locator(mdates.MonthLocator())

ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend()

plt.tight_layout()
plt.show()

print("\nListe des pays à prédire :\n")

for p in pays_predits:
    print(p)
