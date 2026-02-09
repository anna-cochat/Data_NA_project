import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from datetime import datetime
import patsy
import statsmodels.formula.api as smf
from scipy.linalg import svd

def load_mydf():
    df = pd.read_excel("NA.xlsx")
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

Y = pd.to_datetime(
    df_clean["Overshoot Day"],
    errors="coerce"
)

if Y.isna().any():
    raise ValueError(
        f"Overshoot Day n'a pas fonctionné sur {Y.isna().sum()} lignes"
    )

df_clean["Overshoot_Day_DOY"] = Y.dt.dayofyear

drop_cols = [
    "Country",
    "Overshoot Day",
]

df_model = df_clean.drop(columns=drop_cols)

df_model["Income Group"] = pd.Categorical(
    df_model["Income Group"],
    categories=["LI", "LM", "UM", "HI"],
    ordered=True
)

df_model["Quality Score"] = pd.Categorical(
    df_model["Quality Score"],
    categories=["2A", "2B", "2C", "3A"]
)

df_model["Region"] = df_model["Region"].astype("category")

df_1 = pd.get_dummies(
    df_model,
    columns=["Quality Score", "Region", "Income Group"],
    drop_first=True,
)

print("Nombre de pays:", len(df_1))
print("\nColonnes créées:")
print(df_1.filter(regex="Quality Score|Region|Income Group").columns.tolist())
print("\nNA restants:", int(df_1.isna().sum().sum()))


train_df, test_df = train_test_split(
    df_1,
    test_size=0.2,
    random_state=123
)

print("Train lignes:", len(train_df))
print("Test lignes:", len(test_df))

# modèle de base
termst = [
    "SDGi",
    'Q("Life Expectancy")',
    "HDI",
    'Q("Per Capita GDP")',
    'Q("Population (millions)")',
    'Q("Income Group_LI")',
    'Q("Income Group_LM")',
    'Q("Income Group_UM")',
    'Q("Income Group_HI")',
    'Q("Quality Score_2A")',
    'Q("Quality Score_2B")',
    'Q("Quality Score_2C")',
    'Q("Quality Score_3A")',
    'Q("Region_Africa")',
    'Q("Region_Asia-Pacific")',
    'Q("Region_Central America/Caribbean")',
    'Q("Region_EU")',
    'Q("Region_Middle East/Central Asia")',
    'Q("Region_North America")',
    'Q("Region_Other Europe")',
    'Q("Region_South America")',
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
    "Cropland",
    'Q("Grazing land")',
    'Q("Forest land")',
    'Q("Fishing ground")',
    "BuiltUp_Biocapacity",
    'Q("Ecological (Deficit) or Reserve")',
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



