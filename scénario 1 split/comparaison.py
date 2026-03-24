import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


y_var = "Overshoot_Day_DOY"

splits = [70, 80, 90]




def backward_stepwise_aic(X, y):

    X = sm.add_constant(X)

    variables = list(X.columns)
    variables.remove("const")

    best_vars = variables.copy()

    while True:

        X_model = sm.add_constant(X[best_vars])
        model = sm.OLS(y, X_model).fit()

        aic_current = model.aic

        aic_list = []
        var_list = []

        for var in best_vars:

            vars_test = best_vars.copy()
            vars_test.remove(var)

            X_test = sm.add_constant(X[vars_test])
            model_test = sm.OLS(y, X_test).fit()

            aic_list.append(model_test.aic)
            var_list.append(var)

        min_aic = min(aic_list)

        if min_aic < aic_current:
            worst_var = var_list[aic_list.index(min_aic)]
            best_vars.remove(worst_var)
        else:
            break

    return best_vars




all_results = []




for s in splits:

    print("\n======================")
    print("SPLIT", s)
    print("======================")

    train = pd.read_csv(f"train{s}.csv", index_col=0)
    test = pd.read_csv(f"test{s}.csv", index_col=0)

    X_train = train.drop(columns=[y_var])
    y_train = train[y_var]

    X_test = test.drop(columns=[y_var])
    y_test = test[y_var]

    n_train = len(train)
    n_test = len(test)

    print("n train =", n_train)
    print("n test =", n_test)

    

    methods = {}

    

    methods["varcompletes"] = X_train.columns.tolist()

    

    step_vars = backward_stepwise_aic(X_train, y_train)
    methods["stepwise"] = step_vars

    

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso = LassoCV(cv=5, random_state=123, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)

    coef = pd.Series(lasso.coef_, index=X_train.columns)
    lasso_vars = coef[coef != 0].index.tolist()

    methods["lasso"] = lasso_vars


    

    for method_name, vars_sel in methods.items():

        print("\n---", method_name, "---")
        print("n vars =", len(vars_sel))

        X_train_sel = X_train[vars_sel]
        X_test_sel = X_test[vars_sel]

        models = {
            "regression": LinearRegression(),
            "rf": RandomForestRegressor(
                n_estimators=500,
                random_state=123
            ),
            "gb": GradientBoostingRegressor(
                random_state=123
            )
        }

        for model_name, model in models.items():

            model.fit(X_train_sel, y_train)

            pred = model.predict(X_test_sel)

            rmse = np.sqrt(
                mean_squared_error(y_test, pred)
            )

            all_results.append({
                "split": s,
                "methode": method_name,
                "modele": model_name,
                "rmse": rmse,
                "n_vars": len(vars_sel),
                "n_train": n_train,
                "n_test": n_test
            })


df_results = pd.DataFrame(all_results)

df_results = df_results.sort_values(
    ["split", "methode", "rmse"]
)


print("======================")

print(df_results)


df_results.to_csv(
    "results_all_methods.csv",
    index=False
)