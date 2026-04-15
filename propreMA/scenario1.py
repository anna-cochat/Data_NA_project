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

coords = pd.read_csv("famd_model_coords.csv").set_index("Country")

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