import streamlit as st
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go


st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: white;
        color: black;
    }

    section[data-testid="stSidebar"] {
        background-color: #f0f2f6;
    }

    div[data-testid="stDataFrame"] {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Découpage FAMD + Modèles")


# =========================
# LOAD
# =========================

df_model = pd.read_csv("df_model.csv", index_col=0)
coords = pd.read_csv("famd_model_coords.csv").set_index("Country")

common = df_model.index.intersection(coords.index)

df_model = df_model.loc[common].copy()
coords = coords.loc[common].copy()

y_var = "Overshoot_Day_DOY"

coords["Y_observe"] = df_model[y_var].notna()
coords["Y_manquant"] = df_model[y_var].isna()

coords["cote"] = np.where(
    coords["Dim.1"] < 0,
    "gauche",
    "droite"
)


# =========================
# SIDEBAR
# =========================

p = st.sidebar.slider(
    "Pourcentage train",
    50,
    95,
    70,
    step=5
) / 100

k = st.sidebar.slider(
    "Nombre de voisins",
    3,
    20,
    10
)

SEED = st.sidebar.number_input(
    "Graine aléatoire",
    value=123
)

np.random.seed(SEED)


# =========================
# DISTANCES
# =========================

dim_cols = [c for c in coords.columns if c.startswith("Dim.")]
X = coords[dim_cols].values

dist = pd.DataFrame(
    euclidean_distances(X),
    index=coords.index,
    columns=coords.index
)


bleu_gauche = coords.index[
    (coords["cote"] == "gauche")
    & coords["Y_observe"]
]

bleu_droite = coords.index[
    (coords["cote"] == "droite")
    & coords["Y_observe"]
]


# =========================
# POIDS
# =========================

p_dict = {}

for c in bleu_gauche:

    d = dist.loc[c].sort_values()

    voisins = d.index[1:k+1]

    n_manquant = coords.loc[
        voisins,
        "Y_manquant"
    ].sum()

    p_dict[c] = n_manquant / k


poids = pd.Series(p_dict) + 0.01
proba = poids / poids.sum()


# =========================
# SPLIT
# =========================

n_gauche = math.ceil(len(bleu_gauche) * p)

gauche_tires = np.random.choice(
    bleu_gauche,
    size=n_gauche,
    replace=False,
    p=proba.loc[bleu_gauche].values
)

n_droite = math.ceil(len(bleu_droite) * p)

droite_tires = np.random.choice(
    bleu_droite,
    size=n_droite,
    replace=False
)

train = pd.Index(gauche_tires).union(pd.Index(droite_tires))

test = coords.index[
    coords["Y_observe"]
].difference(train)


coords["cat"] = "Test"

coords.loc[train, "cat"] = "Train"
coords.loc[coords["Y_manquant"], "cat"] = "Manquant"


# =========================
# RÉSUMÉ SPLIT
# =========================

train_gauche = int((coords.loc[train, "cote"] == "gauche").sum())
train_droite = int((coords.loc[train, "cote"] == "droite").sum())

test_gauche = int((coords.loc[test, "cote"] == "gauche").sum())
test_droite = int((coords.loc[test, "cote"] == "droite").sum())

n_manquants = int(coords["Y_manquant"].sum())

resume_split = pd.DataFrame({
    "Ensemble": ["Train", "Test", "Y manquant", "Total observés"],
    "Total": [len(train), len(test), n_manquants, int(coords["Y_observe"].sum())],
    "Gauche": [train_gauche, test_gauche, None, int(len(bleu_gauche))],
    "Droite": [train_droite, test_droite, None, int(len(bleu_droite))]
})

st.subheader("Résumé du découpage")
st.dataframe(resume_split, width="content", hide_index=True)
# =========================
# PLOT
# =========================

fig = go.Figure()

colors = {
    "Train": "blue",
    "Test": "green",
    "Manquant": "red"
}

for c in ["Train", "Test", "Manquant"]:

    idx = coords.index[coords["cat"] == c]

    fig.add_trace(
        go.Scatter(
            x=coords.loc[idx, "Dim.1"],
            y=coords.loc[idx, "Dim.2"],
            mode="markers",
            text=idx,
            name=c,
            marker=dict(color=colors[c])
        )
    )

fig.add_vline(x=0)

fig.update_layout(
    title="Espace FAMD",
    height=600
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# MODELS
# =========================

train_df = df_model.loc[train]
test_df = df_model.loc[test]

X_train = train_df.drop(columns=[y_var])
y_train = train_df[y_var]

X_test = test_df.drop(columns=[y_var])
y_test = test_df[y_var]


def backward_stepwise_aic(X, y):

    X = sm.add_constant(X)

    vars = list(X.columns)
    vars.remove("const")

    best = vars.copy()

    while True:

        model = sm.OLS(y, sm.add_constant(X[best])).fit()
        aic = model.aic

        aics = []

        for v in best:

            tmp = best.copy()
            tmp.remove(v)

            m = sm.OLS(y, sm.add_constant(X[tmp])).fit()
            aics.append((v, m.aic))

        v_min, aic_min = min(aics, key=lambda x: x[1])

        if aic_min < aic:
            best.remove(v_min)
        else:
            break

    return best


methods = {}

methods["sans sélection"] = X_train.columns.tolist()
methods["stepwise"] = backward_stepwise_aic(X_train, y_train)

scaler = StandardScaler()

Xs = scaler.fit_transform(X_train)

lasso = LassoCV(cv=5).fit(Xs, y_train)

coef = pd.Series(lasso.coef_, index=X_train.columns)

methods["lasso"] = coef[coef != 0].index.tolist()


results = []


for m_name, vars_sel in methods.items():

    Xtr = X_train[vars_sel]
    Xte = X_test[vars_sel]

    models = {
        "regression": LinearRegression(),
        "rf": RandomForestRegressor(500),
        "gb": GradientBoostingRegressor()
    }

    for name, model in models.items():

        model.fit(Xtr, y_train)

        p_tr = model.predict(Xtr)
        p_te = model.predict(Xte)

        rmse_tr = np.sqrt(mean_squared_error(y_train, p_tr))
        rmse_te = np.sqrt(mean_squared_error(y_test, p_te))

        results.append({
            "méthode": m_name,
            "modèle": name,
            "rmse_train": rmse_tr,
            "rmse_test": rmse_te,
            "overfit": rmse_te / rmse_tr,
            "nvars": len(vars_sel)
        })


df = pd.DataFrame(results)


st.subheader("Résultats")


method_filter = st.multiselect(
    "Méthode",
    df["méthode"].unique(),
    default=df["méthode"].unique()
)

model_filter = st.multiselect(
    "Modèle",
    df["modèle"].unique(),
    default=df["modèle"].unique()
)

df2 = df[
    df["méthode"].isin(method_filter)
    & df["modèle"].isin(model_filter)
]

st.dataframe(
    df2.sort_values("rmse_test"),
    use_container_width=True
)