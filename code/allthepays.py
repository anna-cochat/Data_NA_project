import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# 1. LOAD + CLEAN FUNCTION
# =====================================================

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


# =====================================================
# 2. RUN ANALYSIS + TXT EXPORT
# =====================================================

df = load_mydf()
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# LOAD DATA
# =====================================================
df = load_mydf()
df["NA_count"] = df.isna().sum(axis=1)

# Numerical vars
numeric_vars = df.select_dtypes(include="number").columns.tolist()
numeric_vars.remove("NA_count")

# NA threshold
threshold = 3

# Missingness blocks
block_prod = [
    "Cropland_Footprint_Production","Grazing_Footprint_Production",
    "Forest_Footprint_Production","Fish_Footprint_Production",
    "BuiltUp_Footprint_Production","Carbon_Footprint_Production"
]

block_cons = [
    "Cropland_Footprint_Consumption","Grazing_Footprint_Consumption",
    "Forest_Footprint_Consumption","Fish_Footprint_Consumption",
    "BuiltUp_Footprint_Consumption","Carbon_Footprint_Consumption"
]

block_bio = [
    "Cropland_Biocapacity","Grazing_Biocapacity","Forest_Biocapacity",
    "Fishing_Biocapacity","BuiltUp_Biocapacity"
]

def compress_missing(vars_list):
    """Compacte les patterns de missingness."""
    output = []

    # 1. blocs footprint
    if all(v in vars_list for v in block_prod):
        output.append("Missing ALL Production Footprint components")
        vars_list = [v for v in vars_list if v not in block_prod]

    if all(v in vars_list for v in block_cons):
        output.append("Missing ALL Consumption Footprint components")
        vars_list = [v for v in vars_list if v not in block_cons]

    if all(v in vars_list for v in block_bio):
        output.append("Missing ALL Biocapacity components")
        vars_list = [v for v in vars_list if v not in block_bio]

    # 2. leftover individual vars
    if vars_list:
        output.append("Missing individual variables: " + ", ".join(vars_list))

    return output


# =====================================================
# WRITE OUTPUT FILE
# =====================================================
with open("missingness_report.txt", "w", encoding="utf-8") as f:

    for var_x in numeric_vars:

        f.write("\n\n" + "="*60 + "\n")
        f.write(f"VARIABLE STUDIED : {var_x}\n")
        f.write("="*60 + "\n\n")

        # Countries dropped
        df_dropped = df[df[var_x].isna()].copy()

        # Countries high NA
        df_highNA = df[df["NA_count"] >= threshold].copy()

        # -------- DROPPED COUNTRIES --------
        f.write("--- COUNTRIES DROPPED (X missing) ---\n\n")
        if df_dropped.empty:
            f.write("None.\n\n")
        else:
            for _, row in df_dropped.sort_values("NA_count", ascending=False).iterrows():
                c = row["Country"]
                missing = row[row.isna()].index.tolist()
                summary = compress_missing(missing)

                f.write(f"{c:25s} | NA={row['NA_count']}\n")
                for line in summary:
                    f.write(f"   → {line}\n")
                f.write("\n")

        # -------- HIGH NA COUNTRIES --------
        f.write("\n--- COUNTRIES WITH HIGH NA COUNT (>=3) ---\n\n")
        for _, row in df_highNA.sort_values("NA_count", ascending=False).iterrows():
            if pd.isna(row[var_x]):  
                continue  # don't include dropped here

            c = row["Country"]
            missing = row[row.isna()].index.tolist()
            summary = compress_missing(missing)

            f.write(f"{c:25s} | X={row[var_x]} | NA={row['NA_count']}\n")
            for line in summary:
                f.write(f"   → {line}\n")
            f.write("\n")

print("📄 missingness_report.txt generated successfully.")


pop_col = None
for c in df.columns:
    if "pop" in c.lower():     # détecte population, population (millions), etc.
        pop_col = c
        break



df_pop = df[~df[pop_col].isna()].copy()

# Trouver les deux pays les plus peuplés
top2 = df_pop.nlargest(2, pop_col)["Country"].tolist()
print("📌 Pays exclus pour l’échelle :", top2)

# Retirer ces deux pays
df_trim = df_pop[~df_pop["Country"].isin(top2)]

# Plot
plt.figure(figsize=(8,6))
plt.scatter(df_trim[pop_col], df_trim["NA_count"], alpha=0.7)

plt.xlabel(pop_col)
plt.ylabel("Nombre de valeurs manquantes (NA_count)")
plt.title("NA_count vs Population (sans les 2 pays les plus peuplés)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.show()


df_no_na_count = df.dropna().shape[0]
print(df_no_na_count)