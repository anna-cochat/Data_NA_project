import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import statsmodels.api as sm
import itertools

# =========================================================
# 1. CHARGEMENT
# =========================================================

df_model = pd.read_csv("df_model.csv", index_col=0)
coords_base = pd.read_csv("famd_model_coords.csv")
coords_base = coords_base.set_index("Country")

common = df_model.index.intersection(coords_base.index)
df_model = df_model.loc[common].copy()
coords_base = coords_base.loc[common].copy()

y_var = "Overshoot_Day_DOY"

# =========================================================
# 2. FLAGS Y OBSERVE / MANQUANT
# =========================================================

coords_base["Y_observe"] = df_model[y_var].notna()
coords_base["Y_manquant"] = df_model[y_var].isna()
coords_base["cote"] = np.where(coords_base["Dim.1"] < 0, "gauche", "droite")

# =========================================================
# 3. MATRICE DE DISTANCES DANS L'ESPACE FAMD
# =========================================================

dim_cols = [c for c in coords_base.columns if c.startswith("Dim.")]
X = coords_base[dim_cols].values

dist = pd.DataFrame(
    euclidean_distances(X),
    index=coords_base.index,
    columns=coords_base.index
)

# =========================================================
# 4. CALCUL DE p_dict
# =========================================================

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
lignes = []

for c in bleu_gauche:
    d = dist.loc[c].sort_values()
    voisins = d.index[1:k+1]
    n_manquant = coords_base.loc[voisins, "Y_manquant"].sum()
    p = n_manquant / k
    p_dict[c] = p
    for r, v in enumerate(voisins, start=1):
        lignes.append({
            "Country": c,
            "Rang_voisin": r,
            "Voisin": v,
            "Distance": dist.loc[c, v],
            "Voisin_manquant": coords_base.loc[v, "Y_manquant"],
            "p": p
        })

table_voisins = pd.DataFrame(lignes)

# =========================================================
# 5. FONCTION STEPWISE
# =========================================================

def backward_stepwise_aic(X, y, verbose=False):
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
                results.append((trial_model.aic, var, trial_model, trial_vars))
            except Exception:
                continue

        if not results:
            break

        best_aic, removed_var, best_model, best_vars = min(results, key=lambda x: x[0])

        if best_aic < current_aic:
            if verbose:
                print(f"  Suppression de '{removed_var}' (AIC {current_aic:.2f} -> {best_aic:.2f})")
            remaining = best_vars
            current_aic = best_aic
            model = best_model
        else:
            break

    return model, remaining

# =========================================================
# 6. PARAMETRES A TESTER
# =========================================================

multiplicateurs     = [1, 2, 3, 4, 5]
pourcentages_droite = [0.80]

# =========================================================
# 7. BOUCLE GRID SEARCH
# =========================================================

resultats = []
total = len(multiplicateurs) * len(pourcentages_droite)
i = 0

for mult, pct in itertools.product(multiplicateurs, pourcentages_droite):

    i += 1
    print(f"[{i}/{total}] Multiplicateur={mult}, Pct_droite={pct:.0%} ...", end=" ")

    rng = np.random.default_rng(123)
    coords = coords_base.copy()
    coords["tirage"] = False

    for c in bleu_gauche:
        p2 = min(1, mult * p_dict[c])
        coords.loc[c, "tirage"] = rng.random() < p2

    n_droite = int(np.ceil(pct * len(bleu_droite)))
    echantillon_droite = rng.choice(bleu_droite, size=n_droite, replace=False)

    selection_gauche = coords.index[
        (coords["cote"] == "gauche") &
        (coords["tirage"]) &
        (coords["Y_observe"])
    ]

    pays_train = sorted(set(selection_gauche).union(set(echantillon_droite)))
    pays_test = coords.index[
        (coords["Y_observe"]) &
        (~coords.index.isin(pays_train))
    ].tolist()

    df_train = df_model.loc[pays_train].dropna()
    df_test  = df_model.loc[pays_test].dropna()

    n_train = len(df_train)
    n_test  = len(df_test)

    if n_train < 5:
        print(f"Train trop petit ({n_train}), ignore.")
        resultats.append({
            "Multiplicateur": mult,
            "Pct_droite": pct,
            "N_train": n_train,
            "N_test": n_test,
            "N_vars": np.nan,
            "RMSE": np.nan
        })
        continue

    try:
        y_train = df_train[y_var]
        X_train = df_train.drop(columns=[y_var])

        final_model, selected_vars = backward_stepwise_aic(X_train, y_train, verbose=False)
        vars_used = [v for v in selected_vars if v != "const"]

        df_test_reg = df_test[df_test.columns.intersection(vars_used + [y_var])].dropna()
        y_test = df_test_reg[y_var]
        X_test = sm.add_constant(df_test_reg[vars_used], has_constant="add")
        y_pred = final_model.predict(X_test)

        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        print(f"RMSE={rmse}")

        resultats.append({
            "Multiplicateur": mult,
            "Pct_droite": pct,
            "N_train": n_train,
            "N_test": n_test,
            "N_vars": len(vars_used),
            "RMSE": rmse
        })

    except Exception as e:
        print(f"Erreur : {e}")
        resultats.append({
            "Multiplicateur": mult,
            "Pct_droite": pct,
            "N_train": n_train,
            "N_test": n_test,
            "N_vars": np.nan,
            "RMSE": np.nan
        })

# =========================================================
# 8. TABLEAU RESULTAT
# =========================================================

df_resultats = pd.DataFrame(resultats).sort_values(
    ["Multiplicateur", "Pct_droite"],
    ascending=[True, True]
)
print("\n=== RESULTATS GRID SEARCH ===\n")
print(df_resultats.to_string(index=False))

df_resultats.to_csv("tableau_parametre_RMSE.csv", index=False)
print("\nExporte dans tableau_parametre_RMSE.csv")

# =========================================================
# 9. MEILLEURE COMBINAISON
# =========================================================

best = df_resultats.dropna(subset=["RMSE"]).sort_values("RMSE").iloc[0]
mult_best = best["Multiplicateur"]
pct_best  = best["Pct_droite"]

print(f"\n>>> Meilleure combinaison : Multiplicateur={mult_best}, "
      f"Pct_droite={pct_best:.0%}, RMSE={best['RMSE']}")

# =========================================================
# 10. RECONSTRUCTION DU MEILLEUR MODELE
# =========================================================

print("\n" + "="*60)
print("RECONSTRUCTION DU MEILLEUR MODELE LINEAIRE")
print("="*60)

rng = np.random.default_rng(123)
coords = coords_base.copy()
coords["tirage"] = False

for c in bleu_gauche:
    p2 = min(1, mult_best * p_dict[c])
    coords.loc[c, "tirage"] = rng.random() < p2

n_droite = int(np.ceil(pct_best * len(bleu_droite)))
echantillon_droite = rng.choice(bleu_droite, size=n_droite, replace=False)

selection_gauche = coords.index[
    (coords["cote"] == "gauche") &
    (coords["tirage"]) &
    (coords["Y_observe"])
]

pays_train_best = sorted(set(selection_gauche).union(set(echantillon_droite)))
pays_test_best  = coords.index[
    (coords["Y_observe"]) &
    (~coords.index.isin(pays_train_best))
].tolist()

df_train_best = df_model.loc[pays_train_best].dropna()
df_test_best  = df_model.loc[pays_test_best].dropna()

y_train_best = df_train_best[y_var]
X_train_best = df_train_best.drop(columns=[y_var])

best_model, selected_vars_best = backward_stepwise_aic(X_train_best, y_train_best, verbose=True)
vars_used_best = [v for v in selected_vars_best if v != "const"]

# =========================================================
# 11. AFFICHAGE COMPLET DU MODELE
# =========================================================

print("\n" + "="*60)
print("RESUME COMPLET DU MODELE (statsmodels summary)")
print("="*60)
print(best_model.summary())

coef_df = pd.DataFrame({
    "Coefficient": best_model.params,
    "Std Error":   best_model.bse,
    "t-stat":      best_model.tvalues,
    "p-value":     best_model.pvalues,
    "IC_inf_95":   best_model.conf_int()[0],
    "IC_sup_95":   best_model.conf_int()[1],
}).drop(index="const", errors="ignore")

coef_df["|t-stat|"] = coef_df["t-stat"].abs()
coef_df = coef_df.sort_values("|t-stat|", ascending=False).drop(columns="|t-stat|")
coef_df = coef_df.round(4)

print(coef_df.to_string())
coef_df.to_csv("meilleur_modele_coefficients.csv")
print("\nCoefficients exportes dans meilleur_modele_coefficients.csv")

df_test_best_reg = df_test_best[
    df_test_best.columns.intersection(vars_used_best + [y_var])
].dropna()

y_test_best = df_test_best_reg[y_var]
X_test_best = sm.add_constant(df_test_best_reg[vars_used_best], has_constant="add")
y_pred_best = best_model.predict(X_test_best)

rmse_best = np.sqrt(mean_squared_error(y_test_best, y_pred_best))
mae_best  = np.mean(np.abs(y_test_best - y_pred_best))

print(f"\n>>> Performance test - RMSE : {rmse_best:.2f} | MAE : {mae_best:.2f}")
print(f">>> R2 train : {best_model.rsquared:.4f} | R2 ajuste : {best_model.rsquared_adj:.4f}")
print(f">>> N train  : {len(df_train_best)} | N test : {len(df_test_best_reg)}")
print(f">>> Variables retenues : {len(vars_used_best)}")
print(f"    {vars_used_best}")

# =========================================================
# 12. LES DEUX GRAPHIQUES SUR LA MEME PAGE
# =========================================================

colors_cat = {"Train": "blue", "Test": "green", "Y manquant": "red"}

coords["categorie"] = "Test"
coords.loc[coords.index.isin(pays_train_best), "categorie"] = "Train"
coords.loc[coords["Y_manquant"], "categorie"] = "Y manquant"

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Graphique gauche : Predit vs Observe ---
ax1 = axes[0]
ax1.scatter(y_test_best, y_pred_best,
            alpha=0.8, color="steelblue", edgecolors="white", linewidths=0.5)

lims = [min(y_test_best.min(), y_pred_best.min()) - 5,
        max(y_test_best.max(), y_pred_best.max()) + 5]
ax1.plot(lims, lims, "r--", linewidth=1.5, label="Parfait")
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_xlabel("Observe (DOY)")
ax1.set_ylabel("Predit (DOY)")
ax1.set_title(f"Predit vs Observe - RMSE={rmse_best:.2f}")
ax1.legend()

for country in y_test_best.index:
    ax1.annotate(country,
                 (y_test_best[country], y_pred_best[country]),
                 fontsize=6, alpha=0.7,
                 xytext=(3, 3), textcoords="offset points")

# --- Graphique droite : FAMD ---
ax2 = axes[1]

for g in ["Train", "Test", "Y manquant"]:
    sub = coords[coords["categorie"] == g]
    ax2.scatter(sub["Dim.1"], sub["Dim.2"],
                label=f"{g} (n={len(sub)})",
                color=colors_cat[g], alpha=0.8)

ax2.axhline(0, linestyle="--", color="black")
ax2.axvline(0, linestyle="--", color="black")
ax2.set_xlabel("Dim1")
ax2.set_ylabel("Dim2")
ax2.set_title(
    f"FAMD - meilleure combinaison\n"
    f"(mult={mult_best}, droite={pct_best:.0%}, RMSE={best['RMSE']})"
)
ax2.legend()

plt.suptitle("Diagnostic du meilleur modele lineaire", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("meilleur_modele_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphique sauvegarde dans meilleur_modele_diagnostics.png")
