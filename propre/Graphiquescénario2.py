"""
graph_sorted_predictions.py
----------------------------
Graphique : Prédictions triées de l'Overshoot Day pour tous les pays
Scénario 2 — Random Forest (2018)

Prérequis :
    pip install pandas numpy joblib scikit-learn plotly

Fichiers nécessaires dans le même dossier (ou adapter les chemins) :
    - nettoyage.py          (module de préparation des données)
    - famd_model_coords.csv
    - rf_model.pkl
    - selected_vars.pkl
"""

import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from nettoyage import prepare_data

# Variable cible et couleurs pour les catégories
Y_VAR = "Overshoot_Day_DOY"
COLORS = {"Observé": "#1B5E20", "Manquant": "#4A148C"}


#charge les données préparées et les coordonnées du modèle FAMD
df_imputation, df_model, df_famd_model = prepare_data()
coords = pd.read_csv("famd_model_coords.csv").set_index("Country")

common = df_model.index.intersection(coords.index)
df_model = df_model.loc[common].copy()

rf = joblib.load("rf_model.pkl")
selected_vars = joblib.load("selected_vars.pkl")

df_model["y_pred"] = rf.predict(df_model[selected_vars])
df_model["cat"] = np.where(df_model[Y_VAR].isna(), "Manquant", "Observé")


#Utilitaire : DOY → label de date (1 → "01 Jan", 32 → "01 Feb", etc.)
def doy_to_date(doy):
    try:
        return (
            pd.Timestamp("2018-01-01") + pd.Timedelta(days=int(doy) - 1)
        ).strftime("%d %b")
    except Exception:
        return str(doy)


# Préparation du dataframe de visualisation
df_all_pred = df_model[["y_pred", "cat"]].copy()
df_all_pred.index.name = "Country"
df_all_pred = df_all_pred.sort_values("y_pred")
df_all_pred["y_pred_date"] = df_all_pred["y_pred"].apply(doy_to_date)


# Construction du graphique
fig = go.Figure()

# Ligne de référence : médiane des pays observés
median_obs = df_all_pred.loc[df_all_pred["cat"] == "Observé", "y_pred"].median()
median_date = doy_to_date(median_obs)

fig.add_hline(
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
    fig.add_trace(
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
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Overshoot Day prédit : <b>%{customdata}</b>"
                "<extra></extra>"
            ),
        )
    )

# Repères mensuels sur l'axe Y
month_ticks, month_labels = [], []
for month in range(1, 13):
    doy = (
        pd.Timestamp(f"2018-{month:02d}-01") - pd.Timestamp("2018-01-01")
    ).days + 1
    month_ticks.append(doy)
    month_labels.append(pd.Timestamp(f"2018-{month:02d}-01").strftime("%b"))

fig.update_layout(
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

fig.show()