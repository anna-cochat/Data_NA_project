import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nettoyage import prepare_data

# ─── 1. Ré-entraînement des modèles sur les données originales ───────────────

terms_all = [
    'SDGi', 'Life Expectancy', 'HDI', 'Per Capita GDP', 'Population (millions)',
    'Cropland_Footprint_Production', 'BuiltUp_Footprint_Production',
    'Cropland_Footprint_Consumption', 'Forest_Footprint_Consumption',
    'Fish_Footprint_Consumption', 'Cropland', 'Grazing land',
    'Ecological (Deficit) or Reserve', 'Number of Earths required',
    'Income Group_HI', 'Region_Other Europe'
]
y_var = "Overshoot_Day_DOY"

df_imputation, df_model = prepare_data()

mask = df_model[terms_all + [y_var]].notna().all(axis=1)
X_clean = sm.add_constant(df_model.loc[mask, terms_all])
y_clean = df_model.loc[mask, y_var]

# Modèle OLS
ols_result = sm.OLS(y_clean, X_clean).fit()
print("OLS entraîné ✓")

# Modèle Random Forest
df_rf = df_model.dropna(subset=terms_all + [y_var])
X_train, X_test, y_train, y_test = train_test_split(
    df_rf[terms_all], df_rf[y_var], test_size=0.2, random_state=123
)
rf = RandomForestRegressor(n_estimators=500, random_state=123, n_jobs=-1)
rf.fit(X_train, y_train)
print("Random Forest entraîné ✓")

# ─── 2. Chargement + preprocessing du nouveau CSV ────────────────────────────

df_new_raw = pd.read_csv("semi_Xcomplete_Ymissing.csv")
print(f"\nCSV chargé : {df_new_raw.shape[0]} lignes, {df_new_raw.shape[1]} colonnes")

# Renommage identique à load_mydf()
rename_dict = {
    "actual \nCountry Overshoot Day \n2018": "Overshoot Day",
    "Cropland Footprint":                    "Cropland_Footprint_Production",
    "Grazing Footprint":                     "Grazing_Footprint_Production",
    "Forest Product Footprint":              "Forest_Footprint_Production",
    "Fish Footprint":                        "Fish_Footprint_Production",
    "Built up land":                         "BuiltUp_Footprint_Production",
    "Carbon Footprint":                      "Carbon_Footprint_Production",
    "Cropland Footprint.1":                  "Cropland_Footprint_Consumption",
    "Grazing Footprint.1":                   "Grazing_Footprint_Consumption",
    "Forest Product Footprint.1":            "Forest_Footprint_Consumption",
    "Fish Footprint.1":                      "Fish_Footprint_Consumption",
    "Built up land.1":                       "BuiltUp_Footprint_Consumption",
    "Carbon Footprint.1":                    "Carbon_Footprint_Consumption",
    "Built up land.2":                       "BuiltUp_Biocapacity",
    "Total biocapacity ":                    "Total_Biocapacity",
    "Total Ecological Footprint (Production)":  "Total_Footprint_Production",
    "Total Ecological Footprint (Consumption)": "Total_Footprint_Consumption",
}
df_new_raw = df_new_raw.rename(
    columns={c: rename_dict[c] for c in df_new_raw.columns if c in rename_dict}
)

# Nettoyage numérique
force_text = ["Country", "Region", "Income Group", "Overshoot Day", "Quality Score"]

def clean_numeric(s):
    if s.dtype != object:
        return s
    s = s.astype(str).replace(["-", "--", "", "…"], np.nan)
    s = s.str.replace(r"[,$% ]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

for col in df_new_raw.columns:
    if col not in force_text:
        df_new_raw[col] = clean_numeric(df_new_raw[col])

# Encodage catégories — même ordre que prepare_data()
df_new_raw["Income Group"] = pd.Categorical(
    df_new_raw["Income Group"], categories=["LI", "LM", "UM", "HI"], ordered=True
)
df_new_raw["Quality Score"] = pd.Categorical(
    df_new_raw["Quality Score"], categories=["2A", "2B", "2C", "3A"], ordered=True
)
df_new_raw["Region"] = df_new_raw["Region"].astype("category")

# Suppression colonnes inutiles
df_new_proc = df_new_raw.drop(
    columns=[c for c in ["Country", "Overshoot Day", "Overshoot_Day_DOY",
                          "Income_Group_Code"] if c in df_new_raw.columns],
    errors="ignore"
)

# get_dummies identique à prepare_data()
df_new_proc = pd.get_dummies(
    df_new_proc,
    columns=["Income Group", "Quality Score", "Region"],
    drop_first=True,
    dtype="int64"
)

# Alignement sur les colonnes de df_model (ajoute les dummies absentes = 0)
expected_cols = [c for c in df_model.columns if c != y_var]
df_new_aligned = df_new_proc.reindex(columns=expected_cols, fill_value=0)

# Vérification colonnes X
missing = [c for c in terms_all if df_new_aligned[c].isna().all()]
if missing:
    print(f"⚠️  Colonnes entièrement vides : {missing}")

# ─── 3. Prédictions ──────────────────────────────────────────────────────────

mask_new = df_new_aligned[terms_all].notna().all(axis=1)
X_new = df_new_aligned.loc[mask_new, terms_all]
print(f"Lignes prédites : {mask_new.sum()} / {len(df_new_aligned)}")

# OLS
X_new_const = sm.add_constant(X_new, has_constant='add').reindex(
    columns=X_clean.columns, fill_value=0
)
pred_ols = ols_result.predict(X_new_const)

# Random Forest
pred_rf = rf.predict(X_new)

# ─── 4. Export résultats ─────────────────────────────────────────────────────

results = df_new_raw.loc[mask_new].copy()
results["Pred_OLS"]          = pred_ols.values
results["Pred_RF"]           = pred_rf
results["Difference_RF_OLS"] = pred_rf - pred_ols.values

cols_show = (["Country"] if "Country" in results.columns else []) + \
            ["Pred_OLS", "Pred_RF", "Difference_RF_OLS"]
print("\n", results[cols_show].to_string(index=False))

results.to_csv("predictions_OLS_RF.csv", index=False)
print("\nFichier exporté : predictions_OLS_RF.csv")

# ─── 5. Graphique comparatif ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Scatter OLS vs RF ---
ax = axes[0]
ax.scatter(pred_ols, pred_rf, alpha=0.7, color="steelblue",
           edgecolors="white", linewidth=0.5, s=60)
lim = (min(pred_ols.min(), pred_rf.min()) - 5,
       max(pred_ols.max(), pred_rf.max()) + 5)
ax.plot(lim, lim, 'r--', linewidth=1.5, label="Accord parfait (y = x)")
corr = np.corrcoef(pred_ols, pred_rf)[0, 1]
ax.text(0.05, 0.95, f"Corrélation : {corr:.3f}",
        transform=ax.transAxes, fontsize=10, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
ax.set_xlabel("Prédictions OLS", fontsize=11)
ax.set_ylabel("Prédictions Random Forest", fontsize=11)
ax.set_title("OLS vs Random Forest\n(scatter des prédictions)", fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

# --- Histogrammes ---
ax2 = axes[1]
ax2.hist(pred_ols, bins=20, alpha=0.6, color="steelblue",
         label=f"OLS  (moy={pred_ols.mean():.1f})", edgecolor="white")
ax2.hist(pred_rf,  bins=20, alpha=0.6, color="darkorange",
         label=f"RF   (moy={pred_rf.mean():.1f})",  edgecolor="white")
ax2.axvline(pred_ols.mean(), color="steelblue",  linestyle="--", linewidth=1.5)
ax2.axvline(pred_rf.mean(),  color="darkorange", linestyle="--", linewidth=1.5)
ax2.set_xlabel("Overshoot Day prédit (DOY)", fontsize=11)
ax2.set_ylabel("Nombre de pays", fontsize=11)
ax2.set_title("Distribution des prédictions\nOLS vs Random Forest", fontsize=12, fontweight='bold')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle("Prédictions sur jeu de données à Y manquants",
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("comparaison_OLS_RF_predictions.png", dpi=150, bbox_inches='tight')
plt.show()
print("Graphique sauvegardé : comparaison_OLS_RF_predictions.png")

print(f"\n{'='*55}")
print(f"  OLS — Moy: {pred_ols.mean():.1f} | Std: {pred_ols.std():.1f} | "
      f"Min: {pred_ols.min():.1f} | Max: {pred_ols.max():.1f}")
print(f"  RF  — Moy: {pred_rf.mean():.1f} | Std: {pred_rf.std():.1f} | "
      f"Min: {pred_rf.min():.1f} | Max: {pred_rf.max():.1f}")
print(f"  Corrélation OLS/RF : {corr:.3f}")