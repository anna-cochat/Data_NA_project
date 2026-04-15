import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances


def split(df_model, coords, p=0.7, k=10, seed=123):

    np.random.seed(seed)

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

    p_dict = {}

    for c in bleu_gauche:

        d = dist.loc[c].sort_values()
        voisins = d.index[1:k+1]

        n_manquant = coords.loc[voisins, "Y_manquant"].sum()
        p_dict[c] = n_manquant / k

    poids = pd.Series(p_dict) + 0.01
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

    return train, test