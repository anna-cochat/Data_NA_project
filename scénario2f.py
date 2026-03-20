import pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from nettoyage import prepare_data
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

########## chargement des données = utilisation de la fonction de nettoyage dans le document nettoyage.py
with open("step_model.pkl", "rb") as f:
    step_model = pickle.load(f)

with open("selected_vars.pkl", "rb") as f:
    selected_vars = pickle.load(f)

################################

df_imputation, df_model = prepare_data()
vars_no_const = [v for v in selected_vars if v != "const"]

### c'est tout les X sans NA 
X = df_model[vars_no_const].dropna()
X = sm.add_constant(X, has_constant="add")

y_var = "Overshoot_Day_DOY"
y = df_model.loc[X.index, y_var]
y_pred = step_model.predict(X)

plt.figure(figsize=(8, 6))

plt.scatter(y, y_pred, alpha=0.6, label="Prédictions")

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
plt.title("Scénario 1 – Valeurs réelles vs prédictions (y sans NA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

