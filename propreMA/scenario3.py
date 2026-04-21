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