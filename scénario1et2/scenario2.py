import pandas as pd
import numpy as np
import joblib

from nettoyage import prepare_data
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go

y_var = "Overshoot_Day_DOY"

df_imputation, df_model, df_famd_complete, df_famd_model = prepare_data()

coords = pd.read_csv("/Users/admin/Documents/GitHub/Data_NA_project/scénario1et2/famd_model_coords.csv").set_index("Country")

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


# ── Graphique : prédictions triées pour tous les pays ──

df_all_pred = df_model[["y_pred", "cat"]].copy()
df_all_pred.index.name = "Country"
df_all_pred = df_all_pred.sort_values("y_pred")

# Conversion DOY → date 2018
def doy_to_date(doy):
    try:
        return (pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(doy) - 1)).strftime("%d %b")
    except:
        return str(doy)

df_all_pred["y_pred_date"] = df_all_pred["y_pred"].apply(doy_to_date)

COLORS = {"Observé": "#2D6A4F", "Manquant": "#E76F51"}

fig_sorted = go.Figure()

# Ligne de référence : médiane des pays observés
median_obs = df_all_pred.loc[df_all_pred["cat"] == "Observé", "y_pred"].median()
median_date = doy_to_date(median_obs)

fig_sorted.add_hline(
    y=median_obs,
    line_dash="dot",
    line_color="#999",
    line_width=1.2,
    annotation_text=f"Médiane observés : {median_date}",
    annotation_position="top right",
    annotation_font=dict(size=11, color="#999"),
)

for cat in ["Observé", "Manquant"]:
    idx = df_all_pred.index[df_all_pred["cat"] == cat]
    fig_sorted.add_trace(
        go.Scatter(
            x=list(idx),
            y=df_all_pred.loc[idx, "y_pred"],
            mode="markers",
            marker=dict(
                size=9,
                color=COLORS[cat],
                opacity=0.85,
                line=dict(width=0.5, color="white"),
            ),
            name=cat,
            customdata=df_all_pred.loc[idx, "y_pred_date"],
            hovertemplate="<b>%{x}</b><br>Overshoot Day prédit : <b>%{customdata}</b><extra></extra>",
        )
    )

# Tick labels sur l'axe Y : on place des repères mensuels
month_ticks = []
month_labels = []
for month in range(1, 13):
    doy = (pd.Timestamp(f"2018-{month:02d}-01") - pd.Timestamp("2018-01-01")).days + 1
    month_ticks.append(doy)
    month_labels.append(pd.Timestamp(f"2018-{month:02d}-01").strftime("%b"))

fig_sorted.update_layout(
    title=dict(
        text="Prédiction de l'Overshoot Day — Random Forest (2018)",
        font=dict(family="Georgia, serif", size=17, color="#1a1a2e"),
        x=0.04,
    ),
    xaxis=dict(
        title="Pays",
        tickangle=90,
        tickfont=dict(size=9.5, color="#444"),
        showgrid=False,
        linecolor="#CCCCCC",
    ),
    yaxis=dict(
        title="Overshoot Day prédit",
        tickvals=month_ticks,
        ticktext=month_labels,
        showgrid=True,
        gridcolor="#EBEBEB",
        gridwidth=1,
        zeroline=False,
        linecolor="#CCCCCC",
        tickfont=dict(size=11),
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#e0e0e0",
        borderwidth=1,
    ),
    font=dict(family="Georgia, serif", size=13, color="#1a1a2e"),
    paper_bgcolor="white",
    plot_bgcolor="#FAFAF8",
    hovermode="closest",
    height=620,
    margin=dict(t=80, b=130, l=70, r=30),
)

fig_sorted.show()

