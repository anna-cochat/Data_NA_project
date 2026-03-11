import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from nettoyage import prepare_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

terms_all = [
  'SDGi', 'Life Expectancy', 'HDI', 'Per Capita GDP', 'Population (millions)', 'Cropland_Footprint_Production', 'BuiltUp_Footprint_Production', 'Cropland_Footprint_Consumption', 'Forest_Footprint_Consumption', 'Fish_Footprint_Consumption', 'Cropland', 'Grazing land', 'Ecological (Deficit) or Reserve', 'Number of Earths required', 'Income Group_HI', 'Region_Other Europe'
]
df_imputation, df_model = prepare_data()
### j'ai enlevé vars_no_const car je ne sais pas ce que c'est 
# On prend toutes les variables sélectionnées sauf 'const'
X = df_model[terms_all]
# y = colonne cible
y_var = "Overshoot_Day_DOY"
y = df_model[y_var]

mask = df_model[terms_all + [y_var]].notna().all(axis=1)
X_clean = sm.add_constant(df_model.loc[mask, terms_all])
y_clean = df_model.loc[mask, y_var]

result = sm.OLS(y_clean, X_clean).fit()
y_pred_clean = result.predict(X_clean)
################################

mse_step = mean_squared_error(y_clean, y_pred_clean)
print(f"Stepwise Regression MSE: {mse_step:.2f}")
#842.74
plt.figure(figsize=(8, 6))
plt.scatter(y_clean, y_pred_clean, alpha=0.6, label="Prédictions")
min_val = min(y_clean.min(), y_pred_clean.min())
max_val = max(y_clean.max(), y_pred_clean.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label="y = x"
)

plt.xlabel("Valeurs réelles (Overshoot_Day_DOY)")
plt.ylabel("Valeurs prédites")
plt.title("Scénario 1 – Valeurs réelles vs prédictions (y sans NA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#################### 
# ####Random Forest pour comparer les résultats avec le modèle de régression linéaire

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
df_model_rf = df_model.dropna(subset=terms_all + [y_var])
X_rf = df_model_rf[terms_all]
y_rf = df_model_rf[y_var]
#print(len(df_model_rf))
#print(df_model_rf.head())
print("Noms colonnes X_rf:")
print(X_rf.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X_rf,
    y_rf,
    test_size=0.2,
    random_state=123
)

rf = RandomForestRegressor(
    n_estimators=500,
    random_state=123,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label="Prédictions")
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label="y = x"
)

plt.xlabel("Valeurs réelles (Overshoot_Day_DOY)")
plt.ylabel("Valeurs prédites")
plt.title("Scénario 1 – Random Forest – y_test vs y_pred")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# calcul MSE
mse_rf = mean_squared_error(y_test, y_pred)
print(f"Random Forest MSE: ", mse_rf)
# 18.78

#mse_step = mean_squared_error(y, y_predstep)
#print(f"Stepwise Regression MSE: {mse_step:.2f}")
# 20.45
# #montre les variables du randomforest importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_rf.columns
print("Feature importances (Random Forest):")
print("---------------------------------")

for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")
