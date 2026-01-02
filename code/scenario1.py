import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from datetime import datetime
import patsy
from scipy.linalg import svd

def load_mydf():
    df = pd.read_excel("data/toutlespays.xlsx")

    
    rename_dict = {
        "actual \nCountry Overshoot Day \n2018": "Overshoot Day",

        
        "Cropland Footprint": "Cropland_Footprint_Production",
        "Grazing Footprint": "Grazing_Footprint_Production",
        "Forest Product Footprint": "Forest_Footprint_Production",
        "Fish Footprint": "Fish_Footprint_Production",
        "Built up land": "BuiltUp_Footprint_Production",
        "Carbon Footprint": "Carbon_Footprint_Production",

        
        "Cropland Footprint.1": "Cropland_Footprint_Consumption",
        "Grazing Footprint.1": "Grazing_Footprint_Consumption",
        "Forest Product Footprint.1": "Forest_Footprint_Consumption",
        "Fish Footprint.1": "Fish_Footprint_Consumption",
        "Built up land.1": "BuiltUp_Footprint_Consumption",
        "Carbon Footprint.1": "Carbon_Footprint_Consumption",

        
        "Cropland": "Cropland",
        "Grazing land": "Grazing land",
        "Forest land": "Forest land",
        "Fishing ground": "Fishing ground",
        "Built up land.2": "BuiltUp_Biocapacity",
        "Total biocapacity ": "Total_Biocapacity",

        
        "Total Ecological Footprint (Production)": "Total_Footprint_Production",
        "Total Ecological Footprint (Consumption)": "Total_Footprint_Consumption",
    }

    df = df.rename(columns={c: rename_dict[c] for c in df.columns if c in rename_dict})

    
    force_text = ["Country", "Region", "Income Group", "Overshoot Day", "Quality Score"]

    
    def clean_numeric(s):
        if s.dtype != object:
            return s
        s = s.astype(str)
        s = s.replace(["-", "--", "", "…"], np.nan)
        s = s.str.replace(r"[,$% ]", "", regex=True)
        return pd.to_numeric(s, errors="coerce")

    for col in df.columns:
        if col not in force_text:
            df[col] = clean_numeric(df[col])

    return df


df = load_mydf()
df_clean = df.dropna().copy()

parsed = pd.to_datetime(
    df_clean["Overshoot Day"],
    errors="coerce"
)

if parsed.isna().any():
    raise ValueError(
        f"Overshoot Day n'a pas fonctionné sur {parsed.isna().sum()} lignes"
    )

df_clean["Overshoot_Day_DOY"] = parsed.dt.dayofyear

drop_cols = [
    "Country",
    "Region",
    "Overshoot Day",
    "Quality Score"
]

df_model = df_clean.drop(columns=drop_cols)

df_model["Income Group"] = pd.Categorical(
    df_model["Income Group"],
    categories=[
        "LI",
        "LM",
        "UM",
        "HI"
    ],
    ordered=True
)

print("Income Group counts:")
print(df_model["Income Group"].value_counts(dropna=False)) 
print(f"Nombre de pays: {len(df_model)}")
print(f"NA restants: {int(df_model.isna().sum().sum())}")


train_df, test_df = train_test_split(
    df_model,
    test_size=0.2,
    random_state=123
)

print("Train lignes:", len(train_df))
print("Test lignes:", len(test_df))


#########################################################
import statsmodels.formula.api as smf


# modèle de base avec tous les termes = pas de plein rang
termst = [
    "SDGi",
    'Q("Life Expectancy")',
    "HDI",
    'Q("Per Capita GDP")',
    'Q("Population (millions)")',
    'C(Q("Income Group"), Treatment(reference="LI"))',
    #"Cropland_Footprint_Production",
    "Grazing_Footprint_Production",
    "Forest_Footprint_Production",
    "Fish_Footprint_Production",
    "BuiltUp_Footprint_Production",
    "Carbon_Footprint_Production",
    #"Cropland_Footprint_Consumption",
    "Grazing_Footprint_Consumption",
    "Forest_Footprint_Consumption",
    "Fish_Footprint_Consumption",
    "BuiltUp_Footprint_Consumption",
    "Carbon_Footprint_Consumption",
    #"Total_Footprint_Production",
    #"Total_Footprint_Consumption",
    #"Total_Biocapacity",
    #"Cropland",
    'Q("Grazing land")',
    'Q("Forest land")',
    'Q("Fishing ground")',
    "BuiltUp_Biocapacity",
    #'Q("Ecological (Deficit) or Reserve")',
    'Q("Number of Earths required")',
    'Q("Number of Countries required")'
]

formulat = "Overshoot_Day_DOY ~ " + " + ".join(termst)

print("Fitting formula:")
modelet = smf.ols(formulat, data=train_df).fit()
print(modelet.summary())

# rang de la matrice
Xt = modelet.model.exog
rankt = np.linalg.matrix_rank(Xt)
print("Rank:", rankt, "Cols:", Xt.shape[1])

#######################################################

# modèle sans les totaux 
terms0 = [
    "SDGi",
    'Q("Life Expectancy")',
    "HDI",
    'Q("Per Capita GDP")',
    'Q("Population (millions)")',
    'C(Q("Income Group"), Treatment(reference="LI"))',
    "Cropland_Footprint_Production",
    "Grazing_Footprint_Production",
    "Forest_Footprint_Production",
    "Fish_Footprint_Production",
    "BuiltUp_Footprint_Production",
    "Carbon_Footprint_Production",
    "Cropland_Footprint_Consumption",
    "Grazing_Footprint_Consumption",
    "Forest_Footprint_Consumption",
    "Fish_Footprint_Consumption",
    "BuiltUp_Footprint_Consumption",
    "Carbon_Footprint_Consumption",
    #"Total_Footprint_Production",
    #"Total_Footprint_Consumption",
    #"Total_Biocapacity",
    "Cropland",
    'Q("Grazing land")',
    'Q("Forest land")',
    'Q("Fishing ground")',
    "BuiltUp_Biocapacity",
    'Q("Ecological (Deficit) or Reserve")',
    'Q("Number of Earths required")',
    'Q("Number of Countries required")'
]

formula0 = "Overshoot_Day_DOY ~ " + " + ".join(terms0)

print("formule :")
modele0 = smf.ols(formula0, data=train_df).fit()
print(modele0.summary())

# rang de la matrice 

X0 = modele0.model.exog
rank0 = np.linalg.matrix_rank(X0)
print("Rank:", rank0, "Cols:", X0.shape[1])

#######################################################

# modele que avec les totaux c'était juste pour avoir une idée = de plein rang
terms1 = [
    #"SDGi",
    #'Q("Life Expectancy")',
    #"HDI",
    #'Q("Per Capita GDP")',
    #'Q("Population (millions)")',
    #'C(Q("Income Group"), Treatment(reference="LI"))',
    #"Cropland_Footprint_Production",
    #"Grazing_Footprint_Production",
    #"Forest_Footprint_Production",
    #"Fish_Footprint_Production",
    #"BuiltUp_Footprint_Production",
    #"Carbon_Footprint_Production",
    #"Cropland_Footprint_Consumption",
    #"Grazing_Footprint_Consumption",
    #"Forest_Footprint_Consumption",
    #"Fish_Footprint_Consumption",
    #"BuiltUp_Footprint_Consumption",
    #"Carbon_Footprint_Consumption",
    "Total_Footprint_Production",
    "Total_Footprint_Consumption",
    "Total_Biocapacity",
    #"Cropland",
    #'Q("Grazing land")',
    #'Q("Forest land")',
    #'Q("Fishing ground")',
    #"BuiltUp_Biocapacity",
    #'Q("Ecological (Deficit) or Reserve")',
    #'Q("Number of Earths required")',
    #'Q("Number of Countries required")'
]

formula1 = "Overshoot_Day_DOY ~ " + " + ".join(terms1)

print("formule :")
modele1 = smf.ols(formula1, data=train_df).fit()
print(modele1.summary())

# rang de la matrice
X1 = modele1.model.exog
rank1 = np.linalg.matrix_rank(X1)
print("Rank:", rank1, "Cols:", X1.shape[1])

#######################################################

# modèle a partir du alias de R = de plein rang
terms2 = [
    "SDGi",
    'Q("Life Expectancy")',
    "HDI",
    'Q("Per Capita GDP")',
    'Q("Population (millions)")',

    "Cropland_Footprint_Production",
    "Grazing_Footprint_Production",
    "Forest_Footprint_Production",
    "Fish_Footprint_Production",
    "BuiltUp_Footprint_Production",

    "Cropland_Footprint_Consumption",
    "Grazing_Footprint_Consumption",
    "Forest_Footprint_Consumption",
    "Fish_Footprint_Consumption",

    'Q("Grazing land")',
    'Q("Forest land")',
    'Q("Fishing ground")',

    'Q("Ecological (Deficit) or Reserve")',
    'Q("Number of Countries required")',

    'C(Q("Income Group"), Treatment(reference="LI"))'
]


formula2 = "Overshoot_Day_DOY ~ " + " + ".join(terms2)

print("formule:")
print(formula2)

modele2 = smf.ols(formula2, data=train_df).fit()
print(modele2.summary())

# rang de la matrice 
X2 = modele2.model.exog
rank2 = np.linalg.matrix_rank(X2)
print("Rank:", rank2, "Cols:", X2.shape[1])

#######################################################


def comblin(formula: str, data: pd.DataFrame):
    """Construit la matrice de conception pour `formula` et signale les dépendances."""
    _, X = patsy.dmatrices(formula, data, return_type="dataframe")

    # tolérance numérique proportionnelle à la taille de la matrice
    svals = np.linalg.svd(X, compute_uv=False)
    tol = np.finfo(float).eps * max(X.shape) * svals[0]
    null_mask = svals < tol

    print(
        f"Rang de la matrice de conception : {np.linalg.matrix_rank(X)}, colonnes : {X.shape[1]}, tolérance : {tol:.2e}"
    )

    if null_mask.any():
        _, _, Vt = svd(X, full_matrices=False)
        print("Dépendances linéaires détectées (vecteurs de base du noyau) :")
        for v in Vt[null_mask]:
            print(pd.Series(v, index=X.columns))
    else:
        print("Aucune dépendance linéaire exacte détectée.")

    return {
        "rang": int(np.linalg.matrix_rank(X)),
        "nb_colonnes": int(X.shape[1]),
        "tolerance": float(tol),
        "masque_noyau": null_mask,
        "colonnes": X.columns,
    }

comb2 = comblin(formulat, train_df)




# test ex pour le modele 2

from statsmodels.stats.diagnostic import het_breuschpagan

resid2 = modele2.resid
exog2 = modele2.model.exog

bp = het_breuschpagan(resid2, exog2)

print("Breusch–Pagan test")
print(f"Statistique de Breusch Pagan   : {bp[0]:.3f}")
print(f"P-valeur: {bp[1]:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
plt.scatter(modele2.fittedvalues, resid2, alpha=0.7)
plt.axhline(0, color="red")
plt.xlabel("Valeurs ajustées")
plt.ylabel("Résidus")
plt.title("Résidus vs valeurs ajustées (modèle2)")
plt.tight_layout()
plt.show()
