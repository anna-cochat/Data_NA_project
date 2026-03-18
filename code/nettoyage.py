import pandas as pd
import numpy as np


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
        df["Overshoot Day"],
        errors="coerce"
    ).dt.dayofyear

    df = df.set_index("Country")

    df = df.drop(columns=["Overshoot Day"])

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

    df["Quality_Score_Code"] = (
        df["Quality Score"]
        .cat.codes
        .replace(-1, np.nan)
    )

    # Region: NOMINAL
    df["Region"] = df["Region"].astype("category")



    df_famd_complete = df.copy()
    df_famd_complete = df_famd_complete.reset_index()
    


    df_famd_model = df.drop(columns=[
        "Ecological (Deficit) or Reserve",
        "Number of Earths required",
        "Number of Countries required"
    ]).copy()

    df_famd_model = df_famd_model.reset_index()
    
    df_imputation = df.copy()
    
    
    df_model = pd.get_dummies(
        df,
        columns=["Income Group", "Quality Score", "Region"],
        drop_first=True,
        dtype="int64"
    )

    df_model = df_model.drop(columns=[
        "Income_Group_Code",
        "Quality_Score_Code",
        "Ecological (Deficit) or Reserve",
        "Number of Earths required",
        "Number of Countries required"
    ])

    return df_imputation, df_model, df_famd_complete, df_famd_model


def describe_df(df, name):

    cols = df.columns
    real_cols = list(cols)

# remove Country if present
    if "Country" in real_cols:
        real_cols.remove("Country")

# remove dummies
    real_cols = [
    c for c in real_cols
    if not (
        c.startswith("Income Group_")
        or c.startswith("Quality Score_")
        or c.startswith("Region_")
    )
    ]

# remove codes if categorical exists
    if "Income Group" in real_cols and "Income_Group_Code" in real_cols:
        real_cols.remove("Income_Group_Code")

    if "Quality Score" in real_cols and "Quality_Score_Code" in real_cols:
        real_cols.remove("Quality_Score_Code")

    n_real_vars = len(real_cols)
    has_dummies = any(
        c.startswith("Income Group_")
        or c.startswith("Quality Score_")
        or c.startswith("Region_")
        for c in cols
    )

    has_codes = (
        "Income_Group_Code" in cols
        or "Quality_Score_Code" in cols
    )

    has_derived = any(
        c in cols for c in [
            "Ecological (Deficit) or Reserve",
            "Number of Earths required",
            "Number of Countries required"
        ]
    )

    has_categorical = any(
        c in cols for c in [
            "Income Group",
            "Quality Score",
            "Region"
        ]
    )

    has_country_index = df.index.name == "Country"


    if "Overshoot_Day_DOY" in cols:

        y = df["Overshoot_Day_DOY"]
        X = df.drop(columns=["Overshoot_Day_DOY"])

        mask_X_complete = X.notna().all(axis=1)
        mask_Y_missing = y.isna()
        mask_Y_present = y.notna()
        mask_X_missing = X.isna().any(axis=1)

        n_Xcomplete_Ypresent = (mask_X_complete & mask_Y_present).sum()
        n_Xcomplete_Ymissing = (mask_X_complete & mask_Y_missing).sum()
        n_Xmissing_Ypresent = (mask_X_missing & mask_Y_present).sum()
        n_Xmissing_Ymissing = (mask_X_missing & mask_Y_missing).sum()

    else:

        n_Xcomplete_Ypresent = np.nan
        n_Xcomplete_Ymissing = np.nan
        n_Xmissing_Ypresent = np.nan
        n_Xmissing_Ymissing = np.nan

    return {
        "DF": name,
        "Pays": df.shape[0],
        "Variables": df.shape[1],
        "Variables_réelles": n_real_vars,
        "Dummies": has_dummies,
        "Numérique": has_codes,
        "Dérivées": has_derived,
        "Catégorielle": has_categorical,
        "Index pays": has_country_index,
        "X&Y presents": n_Xcomplete_Ypresent,
        "X complet & Y manquant": n_Xcomplete_Ymissing,
        "X manquant & Y complet": n_Xmissing_Ypresent,
        "X manquant & Y manquant": n_Xmissing_Ymissing
    }
    
if __name__ == "__main__":

    df_imputation, df_model, df_famd_complete, df_famd_model = prepare_data()


    tables = []

    tables.append(describe_df(df_imputation, "imputation"))
    tables.append(describe_df(df_model, "model"))
    tables.append(describe_df(df_famd_complete, "famd_complete"))
    tables.append(describe_df(df_famd_model, "famd_model"))

    compare_df = pd.DataFrame(tables)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    print("\nComparaison des df \n")
    print(compare_df.to_string())

    df_imputation, df_model, df_famd_complete, df_famd_model = prepare_data()
    df_model.to_csv("df_model.csv")
    df_imputation.to_csv("df_imputation.csv")
    df_famd_complete.to_csv("df_famd_complete.csv", index=False)
    df_famd_model.to_csv("df_famd_model.csv", index=False)