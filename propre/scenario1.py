import pandas as pd
import numpy as np
import math
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from nettoyage import prepare_data
from split import split

seed = 123
np.random.seed(seed)

y_var = "Overshoot_Day_DOY"

df_imputation, df_model, df_famd_model = prepare_data()
df_obs = df_model[df_model[y_var].notna()].copy()

coords = pd.read_csv("/Users/admin/Documents/GitHub/Data_NA_project/propre/famd_model_coords.csv").set_index("Country")

train_idx, test_idx = split(
    df_model,
    coords,
    p=0.7,
    k=10,
    seed=123
)

df_train = df_obs.loc[train_idx]
df_test  = df_obs.loc[test_idx]

X_train = df_train.drop(columns=[y_var])
y_train = df_train[y_var]

X_test = df_test.drop(columns=[y_var])
y_test = df_test[y_var]

def backward_stepwise(X, y):

    remaining = list(X.columns)
    best_aic = np.inf

    while True:
        aic_with_vars = []

        for var in remaining:
            vars_try = [v for v in remaining if v != var]

            X_try = sm.add_constant(X[vars_try])
            model = sm.OLS(y, X_try).fit()

            aic_with_vars.append((model.aic, var, vars_try))

        aic_with_vars.sort()
        best_new_aic, var_removed, best_vars = aic_with_vars[0]

        if best_new_aic < best_aic:
            remaining = best_vars
            best_aic = best_new_aic
        else:
            break

    return remaining

selected_vars = backward_stepwise(X_train, y_train)
if "Fish_Footprint_Consumption" in selected_vars and "Carbon_Footprint_Consumption" in selected_vars:
    selected_vars = [v for v in selected_vars if v not in ["Fish_Footprint_Consumption", "Carbon_Footprint_Consumption", "Total_Footprint_Consumption"]]
    selected_vars.append("Total_Footprint_Consumption")
print("\nVariables sélectionnées")
print(selected_vars)

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=123
)

rf.fit(X_train[selected_vars], y_train)

y_pred_train = rf.predict(X_train[selected_vars])
y_pred_test  = rf.predict(X_test[selected_vars])

rmse_train = math.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = math.sqrt(mean_squared_error(y_test, y_pred_test))

print("\nRMSE train:", rmse_train)
print("RMSE test :", rmse_test)


pred_test_df = pd.DataFrame({
    "Country": X_test.index,
    "y_true": y_test.values,
    "y_pred": y_pred_test
})


print("\nPrédictions\n")
print(pred_test_df.head())

pred_train_df = pd.DataFrame({
    "Country": X_train.index,
    "y_true": y_train.values,
    "y_pred": y_pred_train
})

import joblib
joblib.dump(rf, "rf_model.pkl")
joblib.dump(selected_vars, "selected_vars.pkl")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Tag each country in coords with its split role
coords_plot = coords.copy()

# Countries with observed Y
obs_countries = df_model[df_model[y_var].notna()].index
missing_y_countries = df_model[df_model[y_var].isna()].index

def get_role(country):
    if country in train_idx:
        return "Train"
    elif country in test_idx:
        return "Test"
    elif country in missing_y_countries:
        return "Y manquant"
    else:
        return "Other"

coords_plot["role"] = [get_role(c) for c in coords_plot.index]

colours = {
    "Train":      "#2196F3",
    "Test":       "#FF9800",
    "Y manquant": "#E53935",
    "Other":      "#CCCCCC"
}
sizes = {
    "Train": 40,
    "Test":  40,
    "Y manquant": 60,
    "Other": 20
}

fig, ax = plt.subplots(figsize=(12, 8))

for role, group in coords_plot.groupby("role"):
    ax.scatter(
        group["Dim.1"], group["Dim.2"],
        c=colours[role], s=sizes[role],
        label=role, alpha=0.8, edgecolors="none"
    )

ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
ax.set_xlabel("Dim 1")
ax.set_ylabel("Dim 2")
ax.set_title("FAMD — répartition train / test / Y manquant")
ax.legend()
plt.tight_layout()
plt.savefig("famd_split_plot.png", dpi=150)
plt.show()
print("Saved: famd_split_plot.png")

# Also print counts
print("\nRépartition:")
print(coords_plot["role"].value_counts())
print(f"Train: {len(train_idx)}")
print(f"Test:  {len(test_idx)}")
print(f"Total observed: {len(train_idx) + len(test_idx)}")
dim1 = coords["Dim.1"]

train_left  = [c for c in train_idx if dim1[c] < 0]
train_right = [c for c in train_idx if dim1[c] >= 0]
test_left   = [c for c in test_idx  if dim1[c] < 0]
test_right  = [c for c in test_idx  if dim1[c] >= 0]

print(f"Train — left:  {len(train_left)}, right: {len(train_right)}")
print(f"Test  — left:  {len(test_left)},  right: {len(test_right)}")



from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

results = []

# ── Régression linéaire ─────────────────────────────────────────────────────

# Sans sélection
ols_full = sm.OLS(y_train, sm.add_constant(X_train)).fit()
rmse_tr = math.sqrt(mean_squared_error(y_train, ols_full.predict(sm.add_constant(X_train))))
rmse_te = math.sqrt(mean_squared_error(y_test,  ols_full.predict(sm.add_constant(X_test))))
results.append(("Régression linéaire", "Sans sélection", round(rmse_tr,2), round(rmse_te,2)))

# Stepwise AIC
ols_step_vars = backward_stepwise(X_train, y_train)
ols_step = sm.OLS(y_train, sm.add_constant(X_train[ols_step_vars])).fit()
rmse_tr = math.sqrt(mean_squared_error(y_train, ols_step.predict(sm.add_constant(X_train[ols_step_vars]))))
rmse_te = math.sqrt(mean_squared_error(y_test,  ols_step.predict(sm.add_constant(X_test[ols_step_vars]))))
results.append(("Régression linéaire", "Stepwise AIC", round(rmse_tr,2), round(rmse_te,2)))

# LASSO
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
lasso_cv = LassoCV(cv=5, random_state=123, max_iter=10000).fit(X_train_sc, y_train)
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000).fit(X_train_sc, y_train)
lasso_vars = X_train.columns[lasso.coef_ != 0].tolist()
rmse_tr = math.sqrt(mean_squared_error(y_train, lasso.predict(X_train_sc)))
rmse_te = math.sqrt(mean_squared_error(y_test,  lasso.predict(X_test_sc)))
results.append(("Régression linéaire", "LASSO", round(rmse_tr,2), round(rmse_te,2)))
print(f"LASSO variables retenues : {len(lasso_vars)}")

# ── Forêt aléatoire ─────────────────────────────────────────────────────────

# Sans sélection
rf_full = RandomForestRegressor(n_estimators=500, random_state=123)
rf_full.fit(X_train, y_train)
rmse_tr = math.sqrt(mean_squared_error(y_train, rf_full.predict(X_train)))
rmse_te = math.sqrt(mean_squared_error(y_test,  rf_full.predict(X_test)))
results.append(("Forêt aléatoire", "Sans sélection", round(rmse_tr,2), round(rmse_te,2)))

# Stepwise AIC (déjà calculé : selected_vars)
rmse_tr = math.sqrt(mean_squared_error(y_train, rf.predict(X_train[selected_vars])))
rmse_te = math.sqrt(mean_squared_error(y_test,  rf.predict(X_test[selected_vars])))
results.append(("Forêt aléatoire", "Stepwise AIC", round(rmse_tr,2), round(rmse_te,2)))

# LASSO vars
rf_lasso = RandomForestRegressor(n_estimators=500, random_state=123)
rf_lasso.fit(X_train[lasso_vars], y_train)
rmse_tr = math.sqrt(mean_squared_error(y_train, rf_lasso.predict(X_train[lasso_vars])))
rmse_te = math.sqrt(mean_squared_error(y_test,  rf_lasso.predict(X_test[lasso_vars])))
results.append(("Forêt aléatoire", "LASSO", round(rmse_tr,2), round(rmse_te,2)))

# ── Gradient boosting ───────────────────────────────────────────────────────

# Sans sélection
gb_full = GradientBoostingRegressor(n_estimators=500, random_state=123)
gb_full.fit(X_train, y_train)
rmse_tr = math.sqrt(mean_squared_error(y_train, gb_full.predict(X_train)))
rmse_te = math.sqrt(mean_squared_error(y_test,  gb_full.predict(X_test)))
results.append(("Gradient boosting", "Sans sélection", round(rmse_tr,2), round(rmse_te,2)))

# Stepwise AIC
gb_step = GradientBoostingRegressor(n_estimators=500, random_state=123)
gb_step.fit(X_train[ols_step_vars], y_train)
rmse_tr = math.sqrt(mean_squared_error(y_train, gb_step.predict(X_train[ols_step_vars])))
rmse_te = math.sqrt(mean_squared_error(y_test,  gb_step.predict(X_test[ols_step_vars])))
results.append(("Gradient boosting", "Stepwise AIC", round(rmse_tr,2), round(rmse_te,2)))

# LASSO vars
gb_lasso = GradientBoostingRegressor(n_estimators=500, random_state=123)
gb_lasso.fit(X_train[lasso_vars], y_train)
rmse_tr = math.sqrt(mean_squared_error(y_train, gb_lasso.predict(X_train[lasso_vars])))
rmse_te = math.sqrt(mean_squared_error(y_test,  gb_lasso.predict(X_test[lasso_vars])))
results.append(("Gradient boosting", "LASSO", round(rmse_tr,2), round(rmse_te,2)))

# ── Print résultats ──────────────────────────────────────────────────────────

df_results = pd.DataFrame(results, columns=["Modèle", "Sélection", "RMSE train", "RMSE test"])
print("\n=== TABLEAU COMPARATIF ===")
print(df_results.to_string(index=False))

# ════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE : Prédictions par pays — tous les modèles + RMSE sur le côté
# À coller à la fin de ton script principal
# ════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

def safe_pred(y_pred):
    """Clip les prédictions négatives à 0 — le DOY ne peut pas être négatif."""
    return np.clip(y_pred, 0, None)

# ── 1. Collecter toutes les prédictions sur le jeu de TEST ──────────────────

pred_ols_full = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(ols_full.predict(sm.add_constant(X_test))),
    "Modèle":    "Régression linéaire",
    "Sélection": "Sans sélection",
})

pred_ols_step = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(ols_step.predict(sm.add_constant(X_test[ols_step_vars]))),
    "Modèle":    "Régression linéaire",
    "Sélection": "Stepwise AIC",
})

pred_lasso = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(lasso.predict(X_test_sc)),
    "Modèle":    "Régression linéaire",
    "Sélection": "LASSO",
})

pred_rf_full = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(rf_full.predict(X_test)),
    "Modèle":    "Forêt aléatoire",
    "Sélection": "Sans sélection",
})

pred_rf_step = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(rf.predict(X_test[selected_vars])),
    "Modèle":    "Forêt aléatoire",
    "Sélection": "Stepwise AIC",
})

pred_rf_lasso = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(rf_lasso.predict(X_test[lasso_vars])),
    "Modèle":    "Forêt aléatoire",
    "Sélection": "LASSO",
})

pred_gb_full = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(gb_full.predict(X_test)),
    "Modèle":    "Gradient boosting",
    "Sélection": "Sans sélection",
})

pred_gb_step = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(gb_step.predict(X_test[ols_step_vars])),
    "Modèle":    "Gradient boosting",
    "Sélection": "Stepwise AIC",
})

pred_gb_lasso = pd.DataFrame({
    "Country":   X_test.index,
    "y_true":    y_test.values,
    "y_pred":    safe_pred(gb_lasso.predict(X_test[lasso_vars])),
    "Modèle":    "Gradient boosting",
    "Sélection": "LASSO",
})

# ── 2. Assembler et trier les pays par y_true croissant ─────────────────────

all_preds = pd.concat([
    pred_ols_full, pred_ols_step, pred_lasso,
    pred_rf_full,  pred_rf_step,  pred_rf_lasso,
    pred_gb_full,  pred_gb_step,  pred_gb_lasso,
], ignore_index=True)

# Vérification : affiche les prédictions négatives résiduelles s'il en reste
neg = all_preds[all_preds["y_pred"] < 0]
if len(neg) > 0:
    print(f"⚠️  {len(neg)} prédictions négatives détectées et corrigées à 0 :")
    print(neg[["Country", "Modèle", "Sélection", "y_pred"]])
else:
    print("✅ Aucune prédiction négative.")

country_order = (
    all_preds[["Country", "y_true"]]
    .drop_duplicates("Country")
    .sort_values("y_true")["Country"]
    .tolist()
)
x_pos = {c: i for i, c in enumerate(country_order)}
x_vals = list(range(len(country_order)))

# ── 3. Paramètres visuels ────────────────────────────────────────────────────

colours = {
    "Régression linéaire": {
        "Sans sélection": "#185FA5",
        "Stepwise AIC":   "#5B9BD5",
        "LASSO":          "#A8C8EE",
    },
    "Forêt aléatoire": {
        "Sans sélection": "#3B6D11",
        "Stepwise AIC":   "#639922",
        "LASSO":          "#97C459",
    },
    "Gradient boosting": {
        "Sans sélection": "#534AB7",
        "Stepwise AIC":   "#7F77DD",
        "LASSO":          "#B5B0F0",
    },
}
markers_map = {
    "Régression linéaire": "o",
    "Forêt aléatoire":     "s",
    "Gradient boosting":   "^",
}
linestyles = {
    "Sans sélection": "-",
    "Stepwise AIC":   "--",
    "LASSO":          ":",
}

# ── 4. Figure : graphique principal + panneau RMSE ──────────────────────────

fig, (ax_main, ax_rmse) = plt.subplots(
    1, 2,
    figsize=(20, 7),
    gridspec_kw={"width_ratios": [4, 1]},
)

# y_true en noir
y_true_ordered = [
    all_preds[all_preds["Country"] == c]["y_true"].iloc[0]
    for c in country_order
]
ax_main.plot(
    x_vals, y_true_ordered,
    color="black", linewidth=2.5, zorder=5,
    marker="D", markersize=5, label="y_true (observé)",
)

# Une courbe par (modèle, sélection)
for modele, sels in colours.items():
    for sel, col in sels.items():
        subset = all_preds[
            (all_preds["Modèle"] == modele) &
            (all_preds["Sélection"] == sel)
        ].copy()
        subset["x"] = subset["Country"].map(x_pos)
        subset = subset.sort_values("x")

        ax_main.plot(
            subset["x"], subset["y_pred"],
            color=col,
            linewidth=1.5,
            linestyle=linestyles[sel],
            marker=markers_map[modele],
            markersize=5,
            alpha=0.85,
            label=f"{modele} — {sel}",
        )

ax_main.set_xticks(x_vals)
ax_main.set_xticklabels(country_order, rotation=90, fontsize=8)
ax_main.set_ylabel("Overshoot Day (DOY)", fontsize=12)
ax_main.set_xlabel("Pays (triés par y_true croissant)", fontsize=11)
ax_main.set_title("Prédictions par pays — jeu de test", fontsize=13, fontweight="bold")
ax_main.set_ylim(bottom=0)   # force l'axe Y à démarrer à 0
ax_main.grid(axis="y", color="lightgrey", linewidth=0.5)
ax_main.spines[["top", "right"]].set_visible(False)
ax_main.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9, ncol=2)

# ── 5. Panneau RMSE (barres horizontales) ────────────────────────────────────

bar_colors = []
for _, row in df_results.iterrows():
    bar_colors.append(colours[row["Modèle"]][row["Sélection"]])

y_pos = list(range(len(df_results)))

ax_rmse.barh(
    y_pos, df_results["RMSE test"],
    color=bar_colors, edgecolor="white", height=0.65,
)

for i, val in enumerate(df_results["RMSE test"]):
    ax_rmse.text(val + 0.3, i, f"{val:.1f} j", va="center", fontsize=8)

labels_rmse = [
    f"{r['Modèle'].split()[0]}\n{r['Sélection']}"
    for _, r in df_results.iterrows()
]
ax_rmse.set_yticks(y_pos)
ax_rmse.set_yticklabels(labels_rmse, fontsize=7.5)
ax_rmse.set_xlabel("RMSE test (jours)", fontsize=10)
ax_rmse.set_title("RMSE test\npar modèle", fontsize=11, fontweight="bold")
ax_rmse.spines[["top", "right"]].set_visible(False)
ax_rmse.invert_yaxis()

plt.tight_layout()
plt.savefig("graphique_predictions1.png", dpi=150, bbox_inches="tight")
plt.show()
print("Sauvegardé : graphique_predictions1.png")

# ════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE SUPPLÉMENTAIRE : Prédictions par pays — seulement Stepwise AIC
# À rajouter après le graphique précédent
# ════════════════════════════════════════════════════════════════════════════

# ── Filtrer uniquement les prédictions Stepwise AIC ─────────────────────────

aic_preds = all_preds[all_preds["Sélection"] == "Stepwise AIC"].copy()

colours_aic = {
    "Régression linéaire": "#185FA5",
    "Forêt aléatoire":     "#639922",
    "Gradient boosting":   "#7F77DD",
}
markers_aic = {
    "Régression linéaire": "o",
    "Forêt aléatoire":     "s",
    "Gradient boosting":   "^",
}

# RMSE AIC uniquement
df_rmse_aic = df_results[df_results["Sélection"] == "Stepwise AIC"].reset_index(drop=True)

# ── Figure ───────────────────────────────────────────────────────────────────

fig2, (ax2_main, ax2_rmse) = plt.subplots(
    1, 2,
    figsize=(20, 7),
    gridspec_kw={"width_ratios": [4, 1]},
)

# y_true en noir
ax2_main.plot(
    x_vals, y_true_ordered,
    color="black", linewidth=2.5, zorder=5,
    marker="D", markersize=6, label="y_true (observé)",
)

# Une courbe par modèle (AIC uniquement)
for modele, col in colours_aic.items():
    subset = aic_preds[aic_preds["Modèle"] == modele].copy()
    subset["x"] = subset["Country"].map(x_pos)
    subset = subset.sort_values("x")

    # Encadré sur le modèle retenu : Forêt aléatoire
    is_retained = (modele == "Forêt aléatoire")

    ax2_main.plot(
        subset["x"], subset["y_pred"],
        color=col, linewidth=2,
        linestyle="-",
        marker=markers_aic[modele],
        markersize=7 if is_retained else 5,
        alpha=0.9,
        label=f"{modele} — Stepwise AIC{'  ★ modèle retenu' if is_retained else ''}",
        zorder=4 if is_retained else 3,
    )

    # Encadré visible sur chaque point du modèle retenu
    if is_retained:
        ax2_main.plot(
            subset["x"], subset["y_pred"],
            linestyle="none",
            marker=markers_aic[modele],
            markersize=14,
            markerfacecolor="none",
            markeredgecolor=col,
            markeredgewidth=2,
            zorder=5,
        )

ax2_main.set_xticks(x_vals)
ax2_main.set_xticklabels(country_order, rotation=90, fontsize=8)
ax2_main.set_ylabel("Overshoot Day (DOY)", fontsize=12)
ax2_main.set_xlabel("Pays (triés par y_true croissant)", fontsize=11)
ax2_main.set_title(
    "Prédictions par pays — sélection Stepwise AIC uniquement",
    fontsize=13, fontweight="bold",
)
ax2_main.set_ylim(bottom=0)
ax2_main.grid(axis="y", color="lightgrey", linewidth=0.5)
ax2_main.spines[["top", "right"]].set_visible(False)
ax2_main.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.9)

# ── Panneau RMSE AIC (barres horizontales) ───────────────────────────────────

bar_colors_aic = [colours_aic[r["Modèle"]] for _, r in df_rmse_aic.iterrows()]
y_pos_aic = list(range(len(df_rmse_aic)))

# Barres RMSE test (pleines)
ax2_rmse.barh(
    [y - 0.18 for y in y_pos_aic], df_rmse_aic["RMSE test"],
    color=bar_colors_aic, edgecolor="white", height=0.32,
    label="RMSE test",
)
# Barres RMSE train (hachurées)
ax2_rmse.barh(
    [y + 0.18 for y in y_pos_aic], df_rmse_aic["RMSE train"],
    color=bar_colors_aic, edgecolor="white", height=0.32,
    alpha=0.4, hatch="///",
    label="RMSE train",
)

# Valeurs
for i, (_, row) in enumerate(df_rmse_aic.iterrows()):
    ax2_rmse.text(row["RMSE test"]  + 0.2, i - 0.18, f"{row['RMSE test']:.1f} j",  va="center", fontsize=8)
    ax2_rmse.text(row["RMSE train"] + 0.2, i + 0.18, f"{row['RMSE train']:.1f} j", va="center", fontsize=8, alpha=0.7)

ax2_rmse.set_yticks(y_pos_aic)
ax2_rmse.set_yticklabels(
    [r["Modèle"] for _, r in df_rmse_aic.iterrows()],
    fontsize=9,
)
ax2_rmse.set_xlabel("RMSE (jours)", fontsize=10)
ax2_rmse.set_title("RMSE — AIC\ntest vs train", fontsize=11, fontweight="bold")
ax2_rmse.spines[["top", "right"]].set_visible(False)
ax2_rmse.invert_yaxis()
ax2_rmse.legend(fontsize=8, frameon=False, loc="lower right")

plt.tight_layout()
plt.savefig("graphique_aic.png", dpi=150, bbox_inches="tight")
plt.show()
print("Sauvegardé : graphique_aic.png")