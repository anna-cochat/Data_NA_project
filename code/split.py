import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go
from plotly.subplots import make_subplots

p = 0.8
k = 10
SEED = 123

np.random.seed(SEED)

df_model = pd.read_csv("df_model.csv", index_col=0)

coords = pd.read_csv("famd_model_coords.csv")
coords = coords.set_index("Country")

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


dim_cols = [c for c in coords.columns if c.startswith("Dim.")]

X = coords[dim_cols].values

dist = pd.DataFrame(
    euclidean_distances(X),
    index=coords.index,
    columns=coords.index
)


bleu_gauche = coords.index[
    (coords["cote"] == "gauche") &
    (coords["Y_observe"])
]

bleu_droite = coords.index[
    (coords["cote"] == "droite") &
    (coords["Y_observe"])
]

print("Complets gauche :", len(bleu_gauche))
print("Complets droite :", len(bleu_droite))
print("Manquants :", coords["Y_manquant"].sum())
print("Total :", len(coords))


def make_scatter(index, color, name):

    return go.Scatter(
        x=coords.loc[index, "Dim.1"],
        y=coords.loc[index, "Dim.2"],
        mode="markers",
        marker=dict(color=color, size=7),
        text=index,
        hovertemplate="<b>%{text}</b><br>Dim1=%{x:.2f}<br>Dim2=%{y:.2f}<extra></extra>",
        name=name
    )


fig = make_subplots(rows=1, cols=1)

fig.add_trace(
    make_scatter(
        coords.index[coords["Y_observe"]],
        "steelblue",
        "Observés"
    )
)

fig.add_trace(
    make_scatter(
        coords.index[coords["Y_manquant"]],
        "red",
        "Manquants"
    )
)

fig.add_vline(x=0, line_dash="dash")

fig.update_layout(
    title="Espace FAMD",
    width=600,
    height=500
)

fig.show()


p_dict = {}

for c in bleu_gauche:

    d = dist.loc[c].sort_values()

    voisins = d.index[1:k+1]

    n_manquant = coords.loc[
        voisins,
        "Y_manquant"
    ].sum()

    p_dict[c] = n_manquant / k


poids = pd.Series(p_dict)

poids = poids + 0.01

proba = poids / poids.sum()


n_gauche = math.ceil(len(bleu_gauche) * p)

gauche_tires = pd.Index(
    np.random.choice(
        bleu_gauche,
        size=n_gauche,
        replace=False,
        p=proba.loc[bleu_gauche].values
    )
)


n_droite = math.ceil(len(bleu_droite) * p)

droite_tires = pd.Index(
    np.random.choice(
        bleu_droite,
        size=n_droite,
        replace=False
    )
)


train = gauche_tires.union(droite_tires)

test = coords.index[
    coords["Y_observe"]
].difference(train)


train_gauche = sum(
    coords.loc[train, "cote"] == "gauche"
)

train_droite = sum(
    coords.loc[train, "cote"] == "droite"
)

test_gauche = sum(
    coords.loc[test, "cote"] == "gauche"
)

test_droite = sum(
    coords.loc[test, "cote"] == "droite"
)


print()
print("TRAIN total =", len(train))
print("  gauche =", train_gauche)
print("  droite =", train_droite)

print("TEST total =", len(test))
print("  gauche =", test_gauche)
print("  droite =", test_droite)


coords["categorie"] = "Test"

coords.loc[train, "categorie"] = "Train"

coords.loc[
    coords["Y_manquant"],
    "categorie"
] = "Y manquant"


colors = {
    "Train": "blue",
    "Test": "green",
    "Y manquant": "red"
}


fig2 = go.Figure()

for cat in ["Train", "Test", "Y manquant"]:

    idx = coords.index[
        coords["categorie"] == cat
    ]

    fig2.add_trace(
        go.Scatter(
            x=coords.loc[idx, "Dim.1"],
            y=coords.loc[idx, "Dim.2"],
            mode="markers",
            marker=dict(
                color=colors[cat],
                size=8
            ),
            text=idx,
            hovertemplate="<b>%{text}</b><br>Dim1=%{x:.2f}<br>Dim2=%{y:.2f}<extra></extra>",
            name=f"{cat} (n={len(idx)})"
        )
    )

fig2.add_vline(x=0, line_dash="dash")

fig2.update_layout(
    title=f"Split train/test p={p}",
    width=700,
    height=550
)

fig2.show()

df_model.loc[train].to_csv("train90.csv")
df_model.loc[test].to_csv("test90.csv")
