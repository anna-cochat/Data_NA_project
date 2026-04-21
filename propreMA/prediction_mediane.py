import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from nettoyage import prepare_data
from split import split

seed = 123
np.random.seed(seed)

y_var = "Overshoot_Day_DOY"


df_imputation, df_model, df_famd_model = prepare_data()
df_imputed_amputed_mediane=pd.read_csv("df_imputed_amputed_naive.csv",      index_col=0)
df_base= pd.read_csv("df_amputed.csv",      index_col=0)
# on rajoute y
df_imputed_amputed_mediane["Overshoot_Day_DOY"]= df_base["Overshoot_Day_DOY"]


#on encode le df imputé pour avoir les colonnes utilisées par le modèle
df_mediane_encode = pd.get_dummies(
        df_imputed_amputed_mediane,
        columns=["Income Group", "Quality Score", "Region"],
        drop_first=False,
        dtype="int64"
    )

#on prend les pays complets (y observé) pour faire notre nouveau train
df_obs = df_mediane_encode[df_mediane_encode[y_var].notna()].copy()

#on choppe les coordonnées de la nouvelle afdm avec tous les pays
coords = pd.read_csv("coords_afdm.csv").set_index("Country")

#on fait le split avec les anciens+nouveaux pays complets
train_idx, test_idx = split(
    df_obs,
    coords,
    p=0.7,
    k=10,
    seed=123
)

df_train = df_obs.loc[train_idx]
df_test  = df_obs.loc[test_idx]

#on fait le train et le test avec les résultats du split
X_train = df_train.drop(columns=[y_var])
y_train = df_train[y_var]

X_test = df_test.drop(columns=[y_var])
y_test = df_test[y_var]

#on fait le step
def backward_stepwise(X, y):

    remaining = list(X.columns)
    best_aic = np.inf

    while True:
        aic_with_vars = []

        for var in remaining:
            vars_try = [v for v in remaining if v != var]

            X_try = sm.add_constant(X[vars_try])
            model = sm.OLS(y, X_try).fit()

            aic_with_vars.append((model.aic, var, vars_try))

        aic_with_vars.sort()
        best_new_aic, var_removed, best_vars = aic_with_vars[0]

        if best_new_aic < best_aic:
            remaining = best_vars
            best_aic = best_new_aic
        else:
            break

    return remaining

selected_vars = backward_stepwise(X_train, y_train)
if "Fish_Footprint_Consumption" in selected_vars and "Carbon_Footprint_Consumption" in selected_vars:
    selected_vars = [v for v in selected_vars if v not in ["Fish_Footprint_Consumption", "Carbon_Footprint_Consumption", "Total_Footprint_Consumption"]]
    selected_vars.append("Total_Footprint_Consumption")
print("\nVariables sélectionnées")
print(selected_vars)

#on construit le modèle
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=123
)

#on train le modèle en prennant les pays d'entrainement uniquement sur les variables choisies par le split
rf.fit(X_train[selected_vars], y_train)

#on utilise le modèle pour prédire le y de nos train et test
y_pred_train = rf.predict(X_train[selected_vars])
y_pred_test  = rf.predict(X_test[selected_vars])

#on regarde ca que ca donne
rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = math.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nRMSE train:", rmse_train)
print("RMSE test :", rmse_test)


pred_test_df = pd.DataFrame({
    "Country": X_test.index,
    "y_true": y_test.values,
    "y_pred": y_pred_test
})


print("\nPrédictions\n")
print(pred_test_df.head())

pred_train_df = pd.DataFrame({
    "Country": X_train.index,
    "y_true": y_train.values,
    "y_pred": y_pred_train
})


#on vérifie que les colonnes nécessaires pour la prédiction sont bien la puis on construit un df avec tous nos pays et juste les colonnes sélectionnées
missing_cols = [v for v in selected_vars if v not in df_mediane_encode.columns]
if missing_cols:
    print(f"Colonnes manquantes : {missing_cols}")
else:
    print("C'est bon")
    df_pred= df_mediane_encode[selected_vars]


#on fait les prédictions et on les met dans une nouvelle colonne
df_imputed_amputed_mediane["prediction_Overshoot_Day_DOY"] = rf.predict(df_pred)

pays_nan = df_base[df_base["Overshoot_Day_DOY"].isna()].index

#que les pays qui avaient pas de y
df_plot = df_imputed_amputed_mediane.loc[pays_nan]

#on print les prédictions
print("prédictions du modèle : ", df_plot["prediction_Overshoot_Day_DOY"].sort_values())

#on fait la moyenne des prédictions juste pour voir en général combien elles valent
print("moyenne des overshoot day prédits", df_plot["prediction_Overshoot_Day_DOY"].mean())

df_plot["prediction_Overshoot_Day_DOY"].to_csv("predictions_pays_sans_y.csv")

fig = go.Figure()

fig.add_trace(go.Bar(
    x=df_plot.index,
    y=df_plot["prediction_Overshoot_Day_DOY"],
    marker_color="steelblue",
    name="Prédiction Overshoot Day"
))

fig.update_layout(
    title="Prédiction du jour de dépassement pour les pays à valeur manquante",
    xaxis_title="Pays",
    yaxis_title="Overshoot Day (DOY)",
    xaxis_tickangle=-45,
    template="plotly_white",
    height=600
)
fig.write_html("graphique_predictions_mediane.html")
