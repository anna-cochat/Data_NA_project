from nettoyage import prepare_data
import statsmodels.api as sm

# Load data
df_imputation, df_model = prepare_data()

y_var = "Overshoot_Day_DOY"

# Use ALL data (complete cases only)
full_cc = df_model.dropna()

print("Nombre de lignes utilisées (complete cases) :", len(full_cc))

# Separate y and X
y = full_cc[y_var]
X = full_cc.drop(columns=[y_var])

assert X.isna().sum().sum() == 0

# -------------------------------
# Stepwise regression (AIC)
# -------------------------------

def backward_stepwise_aic(X, y):
    X = sm.add_constant(X)
    remaining = list(X.columns)
    current_aic = sm.OLS(y, X[remaining]).fit().aic

    while True:
        aic_candidates = []
        
        for var in remaining:
            if var == "const":
                continue
            trial_vars = [v for v in remaining if v != var]
            model = sm.OLS(y, X[trial_vars]).fit()
            aic_candidates.append((model.aic, var, model, trial_vars))
        
        best_aic, worst_var, best_model, best_vars = min(aic_candidates, key=lambda x: x[0])
        
        if best_aic < current_aic:
            remaining = best_vars
            current_aic = best_aic
            final_model = best_model
        else:
            break

    return final_model, remaining


step_model, selected_vars = backward_stepwise_aic(X, y)

print("\nVariables sélectionnées :")
print(selected_vars)

print("\nRésumé du modèle final :")
print(step_model.summary())


# -------------------------------
# Count selected categorical levels
# -------------------------------

if "Quality Score_2B" in full_cc.columns:
    print("\nQuality Score_2B counts:")
    print(full_cc["Quality Score_2B"].value_counts())

if "Quality Score_2C" in full_cc.columns:
    print("\nQuality Score_2C counts:")
    print(full_cc["Quality Score_2C"].value_counts())