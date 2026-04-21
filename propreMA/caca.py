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
df_imputed_amputed_missforest=pd.read_csv("propreMA/df_imputed_amputed_missforest.csv",      index_col=0)
df_base= pd.read_csv("propreMA/df_amputed.csv",      index_col=0)
# on rajoute y
df_imputed_amputed_missforest["Overshoot_Day_DOY"]= df_base["Overshoot_Day_DOY"]


#on encode le df imputé pour avoir les colonnes utilisées par le modèle
df_missforest_encode = pd.get_dummies(
        df_imputed_amputed_missforest,
        columns=["Income Group", "Quality Score", "Region"],
        drop_first=False,
        dtype="int64"
    )

#on prend les pays complets (y observé) pour faire notre nouveau train
df_obs = df_missforest_encode[df_missforest_encode[y_var].notna()].copy()

#on choppe les coordonnées de la nouvelle afdm avec tous les pays
coords = pd.read_csv("propreMA/coords_afdm.csv").set_index("Country")

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
missing_cols = [v for v in selected_vars if v not in df_missforest_encode.columns]
if missing_cols:
    print(f"Colonnes manquantes : {missing_cols}")
else:
    print("C'est bon")
    df_pred= df_missforest_encode[selected_vars]


#on fait les prédictions et on les met dans une nouvelle colonne
df_imputed_amputed_missforest["prediction_Overshoot_Day_DOY"] = rf.predict(df_pred)

pays_nan = df_base[df_base["Overshoot_Day_DOY"].isna()].index

#que les pays qui avaient pas de y
df_plot = df_imputed_amputed_missforest.loc[pays_nan]

#on print les prédictions
print("prédictions du modèle : ", df_plot["prediction_Overshoot_Day_DOY"].sort_values())

#on fait la moyenne des prédictions juste pour voir en général combien elles valent
print("moyenne des overshoot day prédits", df_plot["prediction_Overshoot_Day_DOY"].mean())

df_plot["prediction_Overshoot_Day_DOY"].to_csv("predictions_pays_sans_y.csv")


#graph
#fig = go.Figure()

#fig.add_trace(go.Bar(
    #x=df_plot.index,
    #y=df_plot["prediction_Overshoot_Day_DOY"],
    #marker_color="steelblue",
    #name="Prédiction Overshoot Day"
#))

#fig.update_layout(
    #title="Prédiction du jour de dépassement pour les pays à valeur manquante",
    #xaxis_title="Pays",
    #yaxis_title="Overshoot Day (DOY)",
    #xaxis_tickangle=-45,
    #template="plotly_white",
    #height=600
#)
#fig.write_html("graphique_predictions_propre.html")


#print(np.ceil(df_plot["prediction_Overshoot_Day_DOY"]).sort_values().to_string())
#print(len(df_plot))

# ---- données observées depuis df_imputation ----
df_obs_plot = df_imputation[df_imputation["Overshoot_Day_DOY"].notna()][["Overshoot_Day_DOY"]].copy()
df_obs_plot["type"] = "Observé"
df_obs_plot = df_obs_plot.rename(columns={"Overshoot_Day_DOY": "DOY"})

# ---- données prédites ----
df_pred_plot = df_plot[["prediction_Overshoot_Day_DOY"]].copy()
df_pred_plot["type"] = "Prédit"
df_pred_plot = df_pred_plot.rename(columns={"prediction_Overshoot_Day_DOY": "DOY"})

# ---- combinaison et tri ----
df_all = pd.concat([df_obs_plot, df_pred_plot]).sort_values("DOY").reset_index()
df_all = df_all.rename(columns={"index": "Country"})

median_obs = df_obs_plot["DOY"].median()

# ---- graphique
color_obs  = "#7fbf7f"
color_pred = "#b39ddb"

fig, ax = plt.subplots(figsize=(18, 6))
fig.patch.set_facecolor("white")      # ← blanc pur
ax.set_facecolor("white")             # ← blanc pur

for _, row in df_all.iterrows():
    color = color_obs if row["type"] == "Observé" else color_pred
    ax.scatter(row["Country"], row["DOY"], color=color, s=25, zorder=3,
               alpha=0.85, linewidths=0)

ax.axhline(y=median_obs, color="#e8a87c", linestyle="--", linewidth=1.2,
           alpha=0.8)

ax.set_xlabel("Pays", fontsize=10, color="#444")
ax.set_ylabel("Jour de dépassement (DOY)", fontsize=10, color="#444")
ax.set_title("Jour de dépassement : valeurs observées et prédites",
             fontsize=12, color="#333", pad=12)

ax.set_xticks(range(len(df_all)))
ax.set_xticklabels(df_all["Country"], rotation=90, fontsize=5.5, color="#555")
ax.tick_params(axis="y", labelsize=9, colors="#555")
ax.tick_params(axis="x", length=0)   # ← supprime les tirets verticaux des xticks

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#ccc")
ax.spines["bottom"].set_color("#ccc")
ax.grid(False)

patch_obs  = mpatches.Patch(color=color_obs,  label="Y observé")
patch_pred = mpatches.Patch(color=color_pred, label="Y manquant (prédit)")
median_line = plt.Line2D([0], [0], color="#e8a87c", linestyle="--",
                          linewidth=1.2,
                          label=f"Médiane observée : {int(median_obs)} jours")
ax.legend(handles=[patch_obs, patch_pred, median_line], fontsize=9,
          framealpha=0.7, edgecolor="#ccc", loc="upper left")

plt.tight_layout()
plt.savefig("overshoot_predictions.png", dpi=180,
            bbox_inches="tight", facecolor="white")  # ← blanc pur ici aussi
plt.show()


# =========================================================
# QUANTILE REGRESSION FOREST — Intervalles de prédiction
# =========================================================
df_missforest_encode["prediction_Overshoot_Day_DOY"]=df_imputed_amputed_missforest["prediction_Overshoot_Day_DOY"] 

y_autre_var="prediction_Overshoot_Day_DOY"

# pip install quantile-forest

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from quantile_forest import RandomForestQuantileRegressor

alpha = 0.10  # intervalle à 90%

# ── 1. Ré-entraîner un QRF avec les mêmes hyperparamètres que votre RF ──
qrf = RandomForestQuantileRegressor(
    n_estimators=rf.n_estimators,
    max_features=rf.max_features,
    min_samples_leaf=rf.min_samples_leaf,
    random_state=42,
    n_jobs=-1
)

# On entraîne uniquement sur les pays avec Y observé
mask_obs = df_missforest_encode[y_autre_var].notna()
X_all = df_missforest_encode[selected_vars]
X_obs = X_all[mask_obs]
y_obs = df_missforest_encode.loc[mask_obs, y_autre_var]

qrf.fit(X_obs, y_obs)

# ── 2. Prédictions quantiles pour TOUS les pays ──────────────
quantiles = [alpha / 2, 0.5, 1 - alpha / 2]  # 5%, 50%, 95%

preds_q = qrf.predict(X_all, quantiles=quantiles)
# shape : (n_pays, 3)

df_missforest_encode["qrf_lower"] = preds_q[:, 0]
df_missforest_encode["qrf_median"] = preds_q[:, 1]
df_missforest_encode["qrf_upper"] = preds_q[:, 2]
df_missforest_encode["qrf_width"] = df_missforest_encode["qrf_upper"] - df_missforest_encode["qrf_lower"]

# ── 3. Table de sortie ───────────────────────────────────────
pred_table = df_missforest_encode[["qrf_lower", "qrf_median", "qrf_upper", "qrf_width", y_autre_var]].copy()
pred_table.columns = ["PI_lower_90", "Pred_median", "PI_upper_90", "PI_width", "Real"]

print("\nPrédictions QRF + Intervalles à 90%:\n")
print(pred_table.round(1).sort_values("Pred_median").to_string())

print(np.mean(pred_table["PI_width"]))