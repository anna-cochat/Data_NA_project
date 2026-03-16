import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

df_model = pd.read_csv("df_model.csv", index_col=0)
coords = pd.read_csv("famd_model_coords.csv")
coords = coords.set_index("Country")

common = df_model.index.intersection(coords.index)

df_model = df_model.loc[common].copy()
coords = coords.loc[common].copy()

y_var = "Overshoot_Day_DOY"

coords["Y_observe"] = df_model[y_var].notna()
coords["Y_manquant"] = df_model[y_var].isna()


coords["cote"] = np.where(coords["Dim.1"] < 0, "gauche", "droite")

dim_cols = [c for c in coords.columns if c.startswith("Dim.")]

X = coords[dim_cols].values

dist = pd.DataFrame(
    euclidean_distances(X),
    index=coords.index,
    columns=coords.index
)

# =========================================================
# 6. CALCUL DE p POUR LES BLEUS A GAUCHE
# =========================================================

k = 10

bleu_gauche = coords.index[
    (coords["cote"] == "gauche") &
    (coords["Y_observe"])
]

p_dict = {}

lignes = []

for c in bleu_gauche:

    d = dist.loc[c].sort_values()

    voisins = d.index[1:k+1]

    n_manquant = coords.loc[voisins, "Y_manquant"].sum()

    p = n_manquant/ k

    p_dict[c] = p

    for r, v in enumerate(voisins, start=1):

        lignes.append({
            "Country": c,
            "Rang_voisin": r,
            "Voisin": v,
            "Distance": dist.loc[c, v],
            "Voisin_manquant": coords.loc[v, "Y_manquant"],
            "p": p
        })

table_voisins = pd.DataFrame(lignes)


# =========================================================
# 7. BERNOULLI SUR LES BLEUS A GAUCHE
# =========================================================

rng = np.random.default_rng(123)

coords["tirage"] = False
coords["p"] = np.nan

for c in bleu_gauche:

    p = p_dict[c]

    coords.loc[c, "p"] = p

    coords.loc[c, "tirage"] = rng.random() < p


# =========================================================
# 8. BLEUS A DROITE → 10% DANS LE TRAIN
# =========================================================

bleu_droite = coords.index[
    (coords["cote"] == "droite") &
    (coords["Y_observe"])
]

n_droite = int(np.ceil(0.10 * len(bleu_droite)))

echantillon_droite= rng.choice(
    bleu_droite,
    size=n_droite,
    replace=False
)


# =========================================================
# 9. TRAIN FINAL
# =========================================================

selection_gauche = coords.index[
    (coords["cote"] == "gauche") &
    (coords["tirage"]) &
    (coords["Y_observe"])
]
pays_train = sorted(
    set(selection_gauche).union(set(echantillon_droite))
)

coords["train"] = coords.index.isin(pays_train)
# =========================================================
# 9bis. GROUPES POUR PLOT
# =========================================================

coords["categorie"] = "Test"

coords.loc[coords["train"], "categorie"] = "Train"

coords.loc[coords["Y_manquant"], "categorie"] = "Y manquant"
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))

colors = {
    "Train": "blue",
    "Test": "green",
    "Y manquant": "red"
}

for g in coords["categorie"].unique():

    sub = coords[coords["categorie"] == g]

    plt.scatter(
        sub["Dim.1"],
        sub["Dim.2"],
        label=g,
        color=colors[g],
        alpha=0.8
    )



plt.axhline(0, linestyle="--", color="black")
plt.axvline(0, linestyle="--", color="black")

plt.xlabel("Dim1")
plt.ylabel("Dim2")

plt.title(" Train / Test / Manquant dans l'espace AFMD")

plt.legend()

plt.show()


# =========================================================
# 10. TEST = BLEUS NON PRIS
# =========================================================

pays_test = coords.index[
    (coords["Y_observe"]) &
    (~coords["train"])
]


# =========================================================
# 11. DATASETS
# =========================================================

df_train = df_model.loc[pays_train]
df_test = df_model.loc[pays_test]


# =========================================================
# 12. RESUME
# =========================================================

print("Total pays :", len(coords))
print("Bleus :", coords["Y_observe"].sum())
print("Rouges :", coords["Y_manquant"].sum())

print("Train :", len(df_train))
print("Test :", len(df_test))


# =========================================================
# 13. EXPORT
# =========================================================

df_train.to_csv("scenario1_train.csv")
df_test.to_csv("scenario1_test.csv")

table_voisins.to_csv("scenario1_neighbors.csv")

pd.Series(pays_train).to_csv(
    "scenario1_train_countries.csv",
    index=False
)

pd.Series(pays_test).to_csv(
    "scenario1_test_countries.csv",
    index=False
)
