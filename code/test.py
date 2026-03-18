from nettoyage import prepare_data
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

df_imputation, df_model = prepare_data()

full_cc = df_model.dropna()

print("Train lignes:", len(full_cc))

y_var = "Overshoot_Day_DOY"

terms_all = [
    'SDGi',
    'Life Expectancy',
    'HDI',
    'Per Capita GDP',
    'Population (millions)',
    'Cropland_Footprint_Production',
    'Grazing_Footprint_Production',
    'Forest_Footprint_Production',
    'Fish_Footprint_Production',
    'BuiltUp_Footprint_Production',
    'Carbon_Footprint_Production',
    'Cropland_Footprint_Consumption',
    'Grazing_Footprint_Consumption',
    'Forest_Footprint_Consumption',
    'Fish_Footprint_Consumption',
    'Carbon_Footprint_Consumption',
    'Cropland',
    'Grazing land',
    'Forest land',
    'Fishing ground',
    'Ecological (Deficit) or Reserve',
    'Number of Earths required',
    'Number of Countries required',
    'Income Group_LM',
    'Income Group_UM',
    'Income Group_HI',
    'Region_Asia-Pacific',
    'Carbon_Footprint_Consumption',
    'Region_Central America/Caribbean',
    'Region_EU',
    'Region_Middle East/Central Asia',
    'Region_North America',
    'Region_Other Europe',
    'Region_South America'
]

y = full_cc[y_var]

valid_terms = [
    c for c in terms_all
    if full_cc[c].nunique() > 1
]

X = full_cc[valid_terms]

assert X.isna().sum().sum() == 0

# stepwise regression

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

print("Variables sélectionnées :")
print(selected_vars)

print(step_model.summary())


########## VARIABLES SELECTIONNEES : ##########

# SDGi
# Life Expectancy
# HDI
# Per Capita GDP
# Population (millions)
# Cropland_Footprint_Production
# BuiltUp_Footprint_Production
# Cropland_Footprint_Consumption
# Forest_Footprint_Consumption
# Fish_Footprint_Consumption
# Cropland
# Grazing land
# Ecological (Deficit) or Reserve
# Number of Earths required
# Income Group_HI
# Region_Other Europe que fais exactement ce code ?