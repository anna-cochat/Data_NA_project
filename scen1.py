import pandas as pd
pays = pd.read_csv("/Users/admin/Documents/GitHub/Data_NA_project/supervised.csv")  # dict: nom_feuille -> DataFrame
pays
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def load_mydf():
    df = pd.read_excel("/Users/admin/Desktop/MIASHS/MasterM1SSD/ProjetNA/NA.xlsx", sheet_name=None)  # dict: nom_feuille -> DataFrame

    df = df["Feuille 1"]
    
    rename_dict = {
        "actual \nCountry Overshoot Day \n2018": "Overshoot Day",
        "Cropland Footprint": "Cropland_Footprint_Production",
        "Cropland Footprint.1": "Cropland_Footprint_Consumption",
        "Cropland": "Cropland_Biocapacity",
        "Grazing Footprint": "Grazing_Footprint_Production",
        "Grazing Footprint.1": "Grazing_Footprint_Consumption",
        "Grazing land": "Grazing_Biocapacity",
        "Forest Product Footprint": "Forest_Footprint_Production",
        "Forest Product Footprint.1": "Forest_Footprint_Consumption",
        "Forest land": "Forest_Biocapacity",
        "Fish Footprint": "Fish_Footprint_Production",
        "Fish Footprint.1": "Fish_Footprint_Consumption",
        "Fishing ground": "Fishing_Biocapacity",
        "Built up land": "BuiltUp_Footprint_Production",
        "Built up land.1": "BuiltUp_Footprint_Consumption",
        "Built up land.2": "BuiltUp_Biocapacity",
        "Carbon Footprint": "Carbon_Footprint_Production",
        "Carbon Footprint.1": "Carbon_Footprint_Consumption",
        "Total Ecological Footprint (Production)": "Total_Footprint_Production",
        "Total Ecological Footprint (Consumption)": "Total_Footprint_Consumption",
        "Total biocapacity": "Total_Biocapacity"
    }

    df = df.rename(columns={c: rename_dict[c] for c in df.columns if c in rename_dict})

    
    force_text = {"Country", "Region", "Income Group", "Overshoot Day", "Quality Score"}

    
    def clean_numeric_series(s):
        """Convert messy strings to floats, ignore non-numeric columns."""
        if s.dtype != object:
            return s

        st = s.astype(str).str.strip()
        st = st.replace(["-", "--", "", "…"], None)

        
        if not st.str.contains(r"\d").any():
            return pd.to_numeric(st, errors="coerce")

        
        st = (
            st.str.replace("\u202f", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("−", "-", regex=False)
            .str.replace("%", "", regex=False)
        )
        return pd.to_numeric(st, errors="coerce")

    
    for col in df.columns:
        if col not in force_text:
            df[col] = clean_numeric_series(df[col])

    return df




df = load_mydf()


def overshoot_to_doy(s):
    """Convert 'Month Day, 1900' to day-of-year (1–365)."""
    if pd.isna(s):
        return None
    dt = pd.to_datetime(s, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.dayofyear


df["Overshoot_Day_DOY"] = df["Overshoot Day"].apply(overshoot_to_doy)



target = "Overshoot_Day_DOY"

df = df[df[target].notna()]
X_cols = [col for col in df.columns if col != target]
df_clean = df.replace("-", np.nan)
df_supervised = df.dropna(subset=X_cols + [target])

df_supervised = df.dropna(axis=0, how="any")

print(df_supervised)

