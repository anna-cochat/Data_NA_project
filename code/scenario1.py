import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD + CLEAN DATA (R-style cleaning, preserve Excel names)
# ============================================================

def load_mydf():
    df = pd.read_excel("data/toutlespays.xlsx")

    # Exact renames but KEEP spaces in biocapacity vars
    rename_dict = {
        "actual \nCountry Overshoot Day \n2018": "Overshoot Day",

        # Production footprints
        "Cropland Footprint": "Cropland_Footprint_Production",
        "Grazing Footprint": "Grazing_Footprint_Production",
        "Forest Product Footprint": "Forest_Footprint_Production",
        "Fish Footprint": "Fish_Footprint_Production",
        "Built up land": "BuiltUp_Footprint_Production",
        "Carbon Footprint": "Carbon_Footprint_Production",

        # Consumption footprints
        "Cropland Footprint.1": "Cropland_Footprint_Consumption",
        "Grazing Footprint.1": "Grazing_Footprint_Consumption",
        "Forest Product Footprint.1": "Forest_Footprint_Consumption",
        "Fish Footprint.1": "Fish_Footprint_Consumption",
        "Built up land.1": "BuiltUp_Footprint_Consumption",
        "Carbon Footprint.1": "Carbon_Footprint_Consumption",

        # Biocapacity (keep spaces)
        "Cropland": "Cropland",
        "Grazing land": "Grazing land",
        "Forest land": "Forest land",
        "Fishing ground": "Fishing ground",
        "Built up land.2": "BuiltUp_Biocapacity",
        "Total biocapacity ": "Total_Biocapacity",

        # Total footprints
        "Total Ecological Footprint (Production)": "Total_Footprint_Production",
        "Total Ecological Footprint (Consumption)": "Total_Footprint_Consumption",
    }

    df = df.rename(columns={c: rename_dict[c] for c in df.columns if c in rename_dict})

    # Columns to keep as text
    force_text = ["Country", "Region", "Income Group", "Overshoot Day", "Quality Score"]

    # Clean numeric columns
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


# Load data
df = load_mydf()


# ============================================================
# 2. Convert Overshoot Day to DOY
# ============================================================

df["Overshoot_Day_DOY"] = pd.to_datetime(df["Overshoot Day"], errors="coerce").dt.dayofyear
df = df[df["Overshoot_Day_DOY"].notna()]  # keep rows with Y


# ============================================================
# 3. Remove rows with ANY missing predictors
# ============================================================

df = df.dropna(axis=0, how="any")


# ============================================================
# 4. Encode Income Group exactly like R (drop first dummy)
# ============================================================

df = pd.get_dummies(df, columns=["Income Group"], drop_first=True)


# ============================================================
# 5. Build X and y EXACT FORMULA (your modele1 + totals)
# ============================================================

model_vars = [
    "SDGi", "Life Expectancy", "HDI", "Per Capita GDP",
    "Population (millions)",

    # Footprints (Prod + Total)
    "Cropland_Footprint_Production", "Grazing_Footprint_Production",
    "Forest_Footprint_Production", "Fish_Footprint_Production",
    "BuiltUp_Footprint_Production", "Total_Footprint_Production",

    # Consumption + Total
    "Cropland_Footprint_Consumption", "Grazing_Footprint_Consumption",
    "Forest_Footprint_Consumption", "Fish_Footprint_Consumption",
    "BuiltUp_Footprint_Consumption", "Total_Footprint_Consumption",

    # Biocapacity (keep the original Excel names)
    "Cropland", "Grazing land", "Forest land", "Fishing ground",
    "BuiltUp_Biocapacity", "Total_Biocapacity",

    # Other predictors
    "Ecological (Deficit) or Reserve",
    "Number of Countries required",

    # Income groups
    "Income Group_LM", "Income Group_UM",
]

target = "Overshoot_Day_DOY"


# ============================================================
# 6. Ensure all variables exist
# ============================================================

for v in model_vars:
    if v not in df.columns:
        raise ValueError(f"Missing variable in dataset: {v}")


X = df[model_vars].copy()
y = df[target].copy()


# ============================================================
# 7. Train/Test Split (80/20)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123
)


# ============================================================
# 8. Fit model EXACTLY like R (detect & drop singularities)
# ============================================================

# First fit: full matrix
X_full = sm.add_constant(X_train)
print("\nColumns with dtype object:")
print(X_full.select_dtypes(include=['object']).columns.tolist())

for c in X_full.select_dtypes(include=['object']).columns:
    print(f"\n==== COLUMN: {c} ====")
    print(X_full[c].unique()[:20])

model_full = sm.OLS(y_train, X_full).fit()

print("\n\n================ FULL MODEL (before removing collinearity) ================")
print(model_full.summary())

# R identifies singular (collinear) predictors by NA coefficients.
# Python prints 0.0000 exactly for them → replicate R's logic:
coef = model_full.params

non_singular = coef[coef != 0].index.tolist()  # drop vars with 0 coef
non_singular_X = [v for v in non_singular if v != "const"]

print("\nVariables KEPT after R-style collinearity removal:")
print(non_singular_X)

# Build reduced X
X_reduced = X_train[non_singular_X]
X_reduced_sm = sm.add_constant(X_reduced)

# FINAL model (R-style)
model_final = sm.OLS(y_train, X_reduced_sm).fit()


# ============================================================
# 9. Output final model exactly like R
# ============================================================

print("\n\n================ FINAL MODEL (R-style singularity removal) ================")
print(model_final.summary())
