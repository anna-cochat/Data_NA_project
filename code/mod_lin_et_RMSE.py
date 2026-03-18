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

    p = n_manquant / k

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

rng = np.random.default_rng(123)

coords["tirage"] = False
coords["p"] = np.nan

for c in bleu_gauche:

    p = p_dict[c]

    # CHANGER LE MULTIPLICATEUR ICI
    p2 = min(1, 5 * p)

    coords.loc[c, "p"] = p2

    coords.loc[c, "tirage"] = rng.random() < p2


bleu_droite = coords.index[
    (coords["cote"] == "droite") &
    (coords["Y_observe"])
]

# IL FAUT CHANGER LE % ICI
n_droite = int(np.ceil(0.70 * len(bleu_droite)))

echantillon_droite = rng.choice(
    bleu_droite,
    size=n_droite,
    replace=False
)


selection_gauche = coords.index[
    (coords["cote"] == "gauche") &
    (coords["tirage"]) &
    (coords["Y_observe"])
]

pays_train = sorted(
    set(selection_gauche).union(set(echantillon_droite))
)

coords["train"] = coords.index.isin(pays_train)

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


pays_test = coords.index[
    (coords["Y_observe"]) &
    (~coords["train"])
]

df_train = df_model.loc[pays_train]
df_test = df_model.loc[pays_test]

print("Total pays :", len(coords))
print("Bleus :", coords["Y_observe"].sum())
print("Rouges :", coords["Y_manquant"].sum())

print("Train :", len(df_train))
print("Test :", len(df_test))

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
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import numpy as np

print("\n=== STEPWISE REGRESSION ===")

y_var = "Overshoot_Day_DOY"



df_train_reg = df_train.dropna()

y_train = df_train_reg[y_var]
X_train = df_train_reg.drop(columns=[y_var])




def backward_stepwise_aic(X, y):

    X = sm.add_constant(X)
    remaining = list(X.columns)

    model = sm.OLS(y, X).fit()
    current_aic = model.aic

    while True:

        results = []

        for var in remaining:

            if var == "const":
                continue

            trial_vars = [v for v in remaining if v != var]

            try:
                trial_model = sm.OLS(y, X[trial_vars]).fit()
                results.append(
                    (trial_model.aic, var, trial_model, trial_vars)
                )
            except:
                continue

        if len(results) == 0:
            break

        best_aic, removed_var, best_model, best_vars = min(
            results,
            key=lambda x: x[0]
        )

        if best_aic < current_aic:

            remaining = best_vars
            current_aic = best_aic
            model = best_model

        else:
            break

    return model, remaining


final_model, selected_vars = backward_stepwise_aic(X_train, y_train)

print("\nSelected vars:")
print(selected_vars)

print("\nModel summary:")
print(final_model.summary())



vars_used = [v for v in selected_vars if v != "const"]

df_test_reg = df_test.dropna()

df_test_reg = df_test_reg[
    df_test_reg.columns.intersection(vars_used + [y_var])
]

y_test = df_test_reg[y_var]
X_test = df_test_reg[vars_used]

X_test = sm.add_constant(X_test, has_constant="add")

y_pred = final_model.predict(X_test)



rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nTest RMSE :", rmse)
print("Test n :", len(y_test))