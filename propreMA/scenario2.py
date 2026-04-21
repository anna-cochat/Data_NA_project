import pandas as pd
import numpy as np
import joblib

from nettoyage import prepare_data
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go

y_var = "Overshoot_Day_DOY"

df_imputation, df_model, df_famd_model = prepare_data()
coords = pd.read_csv("propreMA/famd_model_coords.csv").set_index("Country")

common = df_model.index.intersection(coords.index)

df_model = df_model.loc[common].copy()
coords = coords.loc[common].copy()

rf = joblib.load("rf_model.pkl")
selected_vars = joblib.load("selected_vars.pkl")

df_model["y_pred"] = rf.predict(df_model[selected_vars])

df_model["cat"] = np.where(
    df_model[y_var].isna(),
    "Manquant",
    "Observé"
)

colors = {
    "Observé": "blue",
    "Manquant": "red"
}


dim_cols = [c for c in coords.columns if c.startswith("Dim.")]

dist = pd.DataFrame(
    euclidean_distances(coords[dim_cols]),
    index=coords.index,
    columns=coords.index
)


k = 5
hover_text = []

for country in df_model.index:

    y_real = df_model.loc[country, y_var]
    y_pred = df_model.loc[country, "y_pred"]

    real_txt = "manquant" if pd.isna(y_real) else round(y_real, 1)

    neighbors = dist.loc[country].nsmallest(k + 1).index[1:]

    neigh_txt = ""

    for n in neighbors:
        y_n = df_model.loc[n, y_var]
        y_n_pred = df_model.loc[n, "y_pred"]

        y_n_txt = "manquant" if pd.isna(y_n) else round(y_n, 1)

        neigh_txt += f"{n}: réel={y_n_txt}, préd={round(y_n_pred,1)}<br>"

    txt = (
        f"<b>{country}</b><br>"
        f"Réel: {real_txt}<br>"
        f"Prédit: {round(y_pred,1)}<br><br>"
        f"<b>5 plus proches voisins:</b><br>{neigh_txt}"
    )

    hover_text.append(txt)


fig = go.Figure()

for cat in ["Observé", "Manquant"]:

    idx = df_model.index[df_model["cat"] == cat]

    fig.add_trace(
        go.Scatter(
            x=coords.loc[idx, "Dim.1"],
            y=coords.loc[idx, "Dim.2"],
            mode="markers",
            hoverinfo="text",
            hovertext=[hover_text[df_model.index.get_loc(c)] for c in idx],
            marker=dict(size=8, color=colors[cat]),
            name=cat
        )
    )

fig.add_vline(x=0)

fig.update_layout(
    title="Espace FAMD + prédictions + KNN (RF + Stepwise)",
    hovermode="closest",
    height=650
)

fig.show()

df_obs_plot = df_model[df_model[y_var].notna()].copy()

fig_pr = go.Figure()

fig_pr.add_trace(
    go.Scatter(
        x=df_obs_plot[y_var],
        y=df_obs_plot["y_pred"],
        mode="markers",
        name="points"
    )
)
min_val = min(df_obs_plot[y_var].min(), df_obs_plot["y_pred"].min())
max_val = max(df_obs_plot[y_var].max(), df_obs_plot["y_pred"].max())

fig_pr.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="y = x"
    )
)

fig_pr.update_layout(
    title="Prédit vs Réel",
    xaxis_title="Réel",
    yaxis_title="Prédit",
    height=600
)

fig_pr.show()

imp = pd.Series(rf.feature_importances_, index=selected_vars)
imp = imp.sort_values()

fig_imp = go.Figure()

fig_imp.add_trace(
    go.Bar(
        x=imp.values,
        y=imp.index,
        orientation="h"
    )
)

fig_imp.update_layout(
    title="Importance des variables (Random Forest)",
    height=700
)

fig_imp.show()

k = 5

knn_df = []

for country in df_model.index:

    neighbors = dist.loc[country].nsmallest(k + 1).index[1:]

    
    neigh_obs = [
        df_model.loc[n, y_var]
        for n in neighbors
        if not pd.isna(df_model.loc[n, y_var])
    ]

    if len(neigh_obs) == 0:
        continue

    knn_mean = np.mean(neigh_obs)

    knn_df.append({
        "Country": country,
        "y_pred": df_model.loc[country, "y_pred"],
        "knn_mean": knn_mean
    })

knn_df = pd.DataFrame(knn_df)

fig_knn = go.Figure()

fig_knn.add_trace(
    go.Scatter(
        x=knn_df["knn_mean"],
        y=knn_df["y_pred"],
        mode="markers",
        name="points"
    )
)

min_val = min(knn_df["knn_mean"].min(), knn_df["y_pred"].min())
max_val = max(knn_df["knn_mean"].max(), knn_df["y_pred"].max())

fig_knn.add_trace(
    go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        name="y = x"
    )
)

fig_knn.update_layout(
    title="Cohérence KNN (prédit vs moyenne voisins)",
    xaxis_title="Moyenne voisins (réel)",
    yaxis_title="Prédit",
    height=600
)

fig_knn.show()
# =========================================================
# PRINT PREDICTIONS FOR COUNTRIES WITH MISSING Y
# =========================================================

df_missing_y = df_model[df_model[y_var].isna()].copy()

pred_table = (
    df_missing_y[["y_pred"]]
    .rename(columns={"y_pred": "Predicted_Overshoot_Day_DOY"})
    .sort_values("Predicted_Overshoot_Day_DOY")
)

print("\nPredictions for missing Y:\n")
print(pred_table.round(1).to_string())






# =========================================================
# QUANTILE REGRESSION FOREST — Intervalles de prédiction
# =========================================================
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
mask_obs = df_model[y_var].notna()
X_all = df_model[selected_vars]
X_obs = X_all[mask_obs]
y_obs = df_model.loc[mask_obs, y_var]

qrf.fit(X_obs, y_obs)

# ── 2. Prédictions quantiles pour TOUS les pays ──────────────
quantiles = [alpha / 2, 0.5, 1 - alpha / 2]  # 5%, 50%, 95%

preds_q = qrf.predict(X_all, quantiles=quantiles)
# shape : (n_pays, 3)

df_model["qrf_lower"] = preds_q[:, 0]
df_model["qrf_median"] = preds_q[:, 1]
df_model["qrf_upper"] = preds_q[:, 2]
df_model["qrf_width"] = df_model["qrf_upper"] - df_model["qrf_lower"]

# ── 3. Table de sortie ───────────────────────────────────────
pred_table = df_model[["qrf_lower", "qrf_median", "qrf_upper", "qrf_width", y_var]].copy()
pred_table.columns = ["PI_lower_90", "Pred_median", "PI_upper_90", "PI_width", "Real"]

print("\nPrédictions QRF + Intervalles à 90%:\n")
print(pred_table.round(1).sort_values("Pred_median").to_string())

# ── 4. Graphique : pays avec Y manquant ─────────────────────
df_miss = df_model[df_model[y_var].isna()].copy().sort_values("qrf_median")

fig_pi = go.Figure()

# Barres d'erreur
fig_pi.add_trace(go.Scatter(
    x=df_miss["qrf_median"],
    y=df_miss.index,
    mode="markers",
    marker=dict(color="red", size=7, symbol="diamond"),
    name="Médiane QRF",
    error_x=dict(
        type="data",
        symmetric=False,
        array=(df_miss["qrf_upper"] - df_miss["qrf_median"]).values,
        arrayminus=(df_miss["qrf_median"] - df_miss["qrf_lower"]).values,
        color="rgba(220,50,50,0.35)",
        thickness=2,
        width=5
    ),
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Médiane: %{x:.1f}<br>"
        "IP 90%%: [%{customdata[0]:.1f}, %{customdata[1]:.1f}]<extra></extra>"
    ),
    customdata=df_miss[["qrf_lower", "qrf_upper"]].values
))

fig_pi.update_layout(
    title="QRF — Prédictions + IP 90% (pays à Y manquant)",
    xaxis_title="Overshoot Day (DOY)",
    yaxis_title="Pays",
    height=max(400, len(df_miss) * 24),
    margin=dict(l=160)
)
fig_pi.show()

# ── 5. Couverture empirique (pays observés) ──────────────────
df_obs = df_model[mask_obs].copy()
df_obs["covered"] = (
    (df_obs[y_var] >= df_obs["qrf_lower"]) &
    (df_obs[y_var] <= df_obs["qrf_upper"])
)
coverage = df_obs["covered"].mean()
print(f"\nCouverture empirique (pays observés) : {coverage:.1%}  (cible : {1-alpha:.0%})")

# ── 6. Graphique couverture : prédit vs réel avec IP ────────
df_obs_s = df_obs.sort_values(y_var)

fig_cov = go.Figure()

for cov, color, name in [(True, "steelblue", "Couvert"), (False, "darkorange", "Non couvert")]:
    sub = df_obs_s[df_obs_s["covered"] == cov]
    fig_cov.add_trace(go.Scatter(
        x=sub[y_var],
        y=sub["qrf_median"],
        mode="markers",
        marker=dict(color=color, size=7),
        name=name,
        hovertext=sub.index,
        error_y=dict(
            type="data",
            symmetric=False,
            array=(sub["qrf_upper"] - sub["qrf_median"]).values,
            arrayminus=(sub["qrf_median"] - sub["qrf_lower"]).values,
            color="rgba(70,130,180,0.25)" if cov else "rgba(255,140,0,0.3)",
            thickness=1.5,
            width=4
        )
    ))

mv = [df_obs[y_var].min(), df_obs[y_var].max()]
fig_cov.add_trace(go.Scatter(
    x=mv, y=mv, mode="lines", name="y = x",
    line=dict(color="gray", dash="dash", width=1.5)
))

fig_cov.update_layout(
    title=f"QRF — Couverture IP 90% sur pays observés : {coverage:.1%}",
    xaxis_title="Réel",
    yaxis_title="Médiane QRF",
    height=600
)
fig_cov.show()

# ── 7. Graphique largeur des IP par pays ─────────────────────
df_width = df_model[["qrf_width", "cat"]].copy().sort_values("qrf_width")

fig_w = go.Figure()
for cat, color in [("Observé", "steelblue"), ("Manquant", "red")]:
    sub = df_width[df_width["cat"] == cat]
    fig_w.add_trace(go.Bar(
        x=sub.index,
        y=sub["qrf_width"],
        name=cat,
        marker_color=color,
        opacity=0.75
    ))

fig_w.update_layout(
    title="Largeur des IP 90% par pays (QRF)",
    xaxis_title="Pays",
    yaxis_title="Largeur IP (jours)",
    xaxis_tickangle=-45,
    barmode="overlay",
    height=500
)
fig_w.show()