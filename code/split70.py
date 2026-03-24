import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

df_model = pd.read_csv("df_model.csv", index_col=0)
coords_base = pd.read_csv("famd_model_coords.csv")
coords_base = coords_base.set_index("Country")

common = df_model.index.intersection(coords_base.index)

df_model = df_model.loc[common].copy()
coords_base = coords_base.loc[common].copy()

y_var = "Overshoot_Day_DOY"


coords_base["Y_observe"] = df_model[y_var].notna()
coords_base["Y_manquant"] = df_model[y_var].isna()
coords_base["cote"] = np.where(coords_base["Dim.1"] < 0, "gauche", "droite")


dim_cols = [c for c in coords_base.columns if c.startswith("Dim.")]
X = coords_base[dim_cols].values

dist = pd.DataFrame(
    euclidean_distances(X),
    index=coords_base.index,
    columns=coords_base.index
)


k = 10

bleu_gauche = coords_base.index[
    (coords_base["cote"] == "gauche") &
    (coords_base["Y_observe"])
]

bleu_droite = coords_base.index[
    (coords_base["cote"] == "droite") &
    (coords_base["Y_observe"])
]

p_dict = {}

for c in bleu_gauche:

    d = dist.loc[c].sort_values()

    voisins = d.index[1:k+1]

    n_manquant = coords_base.loc[voisins, "Y_manquant"].sum()

    p = n_manquant / k

    p_dict[c] = p


rng = np.random.default_rng(123)

poids = np.array(list(p_dict.values()))

print("bleu_gauche =", len(poids))
print("poids > 0 =", np.sum(poids > 0))

pct_train = 0.7

coords = coords_base.copy()

poids = []

for c in bleu_gauche:
    poids.append(p_dict[c])

poids = np.array(poids)

poids = poids + 0.01
proba = poids / poids.sum()

n_gauche = int(np.ceil(pct_train * len(bleu_gauche)))

gauche_tires = rng.choice(
    bleu_gauche,
    size=n_gauche,
    replace=False,
    p=proba
)

n_droite = int(np.ceil(pct_train * len(bleu_droite)))

droite_tires = rng.choice(
    bleu_droite,
    size=n_droite,
    replace=False
)

pays_train = sorted(
    set(gauche_tires).union(set(droite_tires))
)

pays_test = coords.index[
    (coords["Y_observe"]) &
    (~coords.index.isin(pays_train))
].tolist()

train_gauche = sum(coords.loc[pays_train, "cote"] == "gauche")
train_droite = sum(coords.loc[pays_train, "cote"] == "droite")

test_gauche = sum(coords.loc[pays_test, "cote"] == "gauche")
test_droite = sum(coords.loc[pays_test, "cote"] == "droite")

print("TRAIN total =", len(pays_train))
print("  gauche =", train_gauche)
print("  droite =", train_droite)

print("TEST total =", len(pays_test))
print("  gauche =", test_gauche)
print("  droite =", test_droite)


coords["categorie"] = "Test"

coords.loc[
    coords.index.isin(pays_train),
    "categorie"
] = "Train"

coords.loc[
    coords["Y_manquant"],
    "categorie"
] = "Y manquant"

couleurs = {
    "Train": "blue",
    "Test": "green",
    "Y manquant": "red"
}

plt.figure(figsize=(6,5))

for g in ["Train","Test","Y manquant"]:

    sub = coords[coords["categorie"] == g]

    plt.scatter(
        sub["Dim.1"],
        sub["Dim.2"],
        color=couleurs[g],
        label=f"{g} (n={len(sub)})",
        alpha=0.8
    )

plt.axhline(0, linestyle="--", color="black")
plt.axvline(0, linestyle="--", color="black")

plt.title(f"pct_train = {pct_train}")

plt.legend()
plt.tight_layout()
plt.show()

train_df = df_model.loc[pays_train]
test_df = df_model.loc[pays_test]

train_df.to_csv("train70.csv")
test_df.to_csv("test70.csv")