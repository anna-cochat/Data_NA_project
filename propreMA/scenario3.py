import pandas as pd
import numpy as np
import joblib

from nettoyage import prepare_data
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go


y_var = "Overshoot_Day_DOY"

df_imputation, df_model, df_famd_model = prepare_data()
df_imputed_amputed_missforest=pd.read_csv("propreMA/df_imputed_amputed_missforest.csv",      index_col=0)
df_base= pd.read_csv("propreMA/df_amputed.csv",      index_col=0)

df_imputed_amputed_missforest["Overshoot_Day_DOY"]= df_base["Overshoot_Day_DOY"]

df_missforest_encode = pd.get_dummies(
        df_imputed_amputed_missforest,
        columns=["Income Group", "Quality Score", "Region"],
        drop_first=False,
        dtype="int64"
    )

print(df_missforest_encode.columns)
rf = joblib.load("rf_model.pkl")
selected_vars = joblib.load("selected_vars.pkl")

missing_cols = [v for v in selected_vars if v not in df_missforest_encode.columns]
if missing_cols:
    print(f"Colonnes manquantes : {missing_cols}")
else:
    print("C'est bon")
    df_pred= df_missforest_encode[selected_vars]


print(df_pred.head(5))


df_imputed_amputed_missforest["prediction_Overshoot_Day_DOY"] = rf.predict(df_pred)

df_missforest_encode["prediction_Overshoot_Day_DOY"]=df_imputed_amputed_missforest["prediction_Overshoot_Day_DOY"] 

print(df_imputed_amputed_missforest["prediction_Overshoot_Day_DOY"].sort_values())


pays_nan = df_base[df_base["Overshoot_Day_DOY"].isna()].index


# Filtrer le df final sur ces pays
df_plot = df_imputed_amputed_missforest.loc[pays_nan]



print("moyenne des overshoot day prédits", df_plot["prediction_Overshoot_Day_DOY"].mean())


# Graphique
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
fig.write_html("graphique_predictions.html")


print(np.ceil(df_plot["prediction_Overshoot_Day_DOY"]).sort_values().to_string())
print(len(df_plot))

#df_imputed_amputed_missforest[["prediction_Overshoot_Day_DOY"]].to_csv("predictions_imputed.csv")


# =========================================================
# QUANTILE REGRESSION FOREST — Intervalles de prédiction
# =========================================================


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