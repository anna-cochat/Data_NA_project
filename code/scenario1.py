import pandas as pd
import matplotlib.pyplot as plt

# prendre le data set drop na sur les X et Y train et test mais pas de validation

def load_mydf():
    df = pd.read_excel("data/toutlespays.xlsx")

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

df_supervised = df.dropna
df_supervised.head(10)