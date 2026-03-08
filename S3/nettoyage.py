import pandas as pd
import numpy as np
import os

def load_mydf():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fichier = os.path.join(script_dir, "..", "NA.xlsx")
    df = pd.read_excel(fichier)

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

if __name__ == "__main__":
    df_imputation, df_model = prepare_data()

    print(df_imputation.head())
    print(df_model.head())
