from nettoyage import prepare_data
import pandas as pd
import numpy as np

df_imputation, df_model, df_famd_model = prepare_data()

y_var = "Overshoot_Day_DOY"

# ─────────────────────────────────────────
# TABLEAU 1 : configurations X/Y
# ─────────────────────────────────────────

# X complet = aucune NA sur les colonnes hors Y
x_cols_imputation = [c for c in df_imputation.columns if c != y_var]

x_complet = df_imputation[x_cols_imputation].notna().all(axis=1)
y_complet  = df_imputation[y_var].notna()

print("=== TABLEAU CONFIGURATIONS X/Y ===")
print(f"X complet  & Y complet  : {(x_complet & y_complet).sum()}")
print(f"X complet  & Y manquant : {(x_complet & ~y_complet).sum()}")
print(f"X complet  total        : {x_complet.sum()}")
print(f"X manquant & Y complet  : {(~x_complet & y_complet).sum()}")
print(f"X manquant & Y manquant : {(~x_complet & ~y_complet).sum()}")
print(f"X manquant total        : {(~x_complet).sum()}")
print(f"Y complet  total        : {y_complet.sum()}")
print(f"Y manquant total        : {(~y_complet).sum()}")
print(f"Total pays              : {len(df_imputation)}")

# ─────────────────────────────────────────
# TABLEAU 2 : structure des dataframes
# ─────────────────────────────────────────

print("\n=== TABLEAU DATAFRAMES ===")

# df_imputation
cat_cols_imp = df_imputation.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols_imp = df_imputation.select_dtypes(include="number").columns.tolist()
# cat num = colonnes _Code
cat_num_imp = [c for c in num_cols_imp if "Code" in c]
# var dérivées = Overshoot_Day_DOY (seule vraie dérivée restante dans ce df)
derives_imp = [y_var]
print(f"\ndf_imputation:")
print(f"  Pays              : {len(df_imputation)}")
print(f"  Variables totales : {df_imputation.shape[1]}")
print(f"  Var réelles num   : {len([c for c in num_cols_imp if c not in cat_num_imp and c not in derives_imp])}")
print(f"  Cat (object/cat)  : {cat_cols_imp}")
print(f"  Cat num (_Code)   : {cat_num_imp}")
print(f"  Dérivées          : {derives_imp}")
print(f"  Index pays        : {df_imputation.index.name}")

# df_model
num_cols_mod = df_model.select_dtypes(include="number").columns.tolist()
cat_cols_mod = df_model.select_dtypes(include=["category", "object"]).columns.tolist()
print(f"\ndf_model:")
print(f"  Pays              : {len(df_model)}")
print(f"  Variables totales : {df_model.shape[1]}")
print(f"  Colonnes num      : {len(num_cols_mod)}")
print(f"  Colonnes cat      : {cat_cols_mod}")
print(f"  Index pays        : {df_model.index.name}")
print(f"  Toutes les colonnes : {list(df_model.columns)}")

# df_famd_model
cat_cols_famd = df_famd_model.select_dtypes(include=["category", "object"]).columns.tolist()
num_cols_famd = df_famd_model.select_dtypes(include="number").columns.tolist()
cat_num_famd  = [c for c in num_cols_famd if "Code" in c]
print(f"\ndf_famd_model:")
print(f"  Pays              : {len(df_famd_model)}")
print(f"  Variables totales : {df_famd_model.shape[1]}")
print(f"  Colonnes num      : {len(num_cols_famd)}")
print(f"  Colonnes cat      : {cat_cols_famd}")
print(f"  Cat num (_Code)   : {cat_num_famd}")
print(f"  Index pays        : {'Country' if 'Country' in df_famd_model.columns else df_famd_model.index.name}")
print(f"  Toutes les colonnes : {list(df_famd_model.columns)}")


from nettoyage import prepare_data
import pandas as pd

df_imputation, df_model, df_famd_model = prepare_data()

print(round(df_imputation.isnull().sum() / len(df_imputation) * 100, 1))
print(df_imputation.dtypes)