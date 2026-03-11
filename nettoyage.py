import pandas as pd
import numpy as np

def load_mydf():
    df = pd.read_excel("/Users/admin/Documents/GitHub/Data_NA_project/NA.xlsx")

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

def prepare_data():
    df = load_mydf()
# DOY
    df["Overshoot_Day_DOY"] = pd.to_datetime(
    df["Overshoot Day"], errors="coerce"
    ).dt.dayofyear

    df = df.drop(columns=["Country", "Overshoot Day"])

# INCOME
    df["Income Group"] = pd.Categorical(
    df["Income Group"],
    categories=["LI", "LM", "UM", "HI"],
    ordered=True
    )

    df["Income_Group_Code"] = (
    df["Income Group"]
    .cat.codes
    .replace(-1, np.nan)
    )

# Quality Score: ORDINALE
    df["Quality Score"] = pd.Categorical(
    df["Quality Score"],
    categories=["2A", "2B", "2C", "3A"],
    ordered=True
    )

# Region: NOMINAL
    df["Region"] = df["Region"].astype("category")

    df_imputation = df.copy()

    df_model = pd.get_dummies(
    df,
    columns=["Income Group", "Quality Score", "Region"],
    drop_first=True,
    dtype="int64"
)
    df_model = df_model.drop(columns=["Income_Group_Code"])
    return df_imputation, df_model

# ca cest juste des verifications pour voir si tout marchait bien tu peux supp

#print(" NA pour Income :")
#print(df_imputation["Income_Group_Code"].isna().sum())

#print("Income valeurs (et NA):")
#print(df_imputation["Income_Group_Code"].value_counts(dropna=False).sort_index())

#print(df_imputation[["Quality Score", "Region"]].dtypes)

#print("NA par colonne:")
#print(df_imputation.isna().sum().sort_values(ascending=False).head(10))

#print("Income dummies:")
#print([c for c in df_model.columns if c.startswith("Income Group_")])

#print("Income_Group_Code dans df_model?")
#print("Income_Group_Code" in df_model.columns)

#print("Nb de na dans df_model:")
#print(df_model.isna().sum().sort_values(ascending=False).head(10))

#print("Lignes imputation:", df_imputation.shape[0])
#print("Lignes modèles:", df_model.shape[0])

#income_missing_idx = df["Income Group"].isna()
#print("Nombre de lignes avec Income Group manquant:", income_missing_idx.sum())

# Vérifier si d'autres variables sont aussi manquantes quand Income Group est manquant

#other_missing = df.loc[income_missing_idx].drop(
    columns=["Income Group", "Income_Group_Code"],
    errors="ignore"
#).isna().any(axis=1)

#print("Quand Income Group est manquant, d'autres variables sont-elles aussi manquantes ?")
#print(other_missing.value_counts())


if __name__ == "__main__":
    df_imputation, df_model = prepare_data()

    print(df_imputation.head())
    print(df_model.head())
