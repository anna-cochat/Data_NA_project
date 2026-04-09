import pandas as pd
import numpy as np
import math
import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from nettoyage import prepare_data
from split70 import split

seed = 123
np.random.seed(seed)

y_var = "Overshoot_Day_DOY"

df_imputation, df_model, df_famd_complete, df_famd_model = prepare_data()

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