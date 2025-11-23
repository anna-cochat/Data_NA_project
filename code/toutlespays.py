import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================
# 1. Load dataset
# =================================================

df = pd.read_excel("toutlespays.xlsx")
df = df[df["Country"].notna()].reset_index(drop=True)

print("Countries loaded:", len(df), "\n")


# =================================================
# 2. Helper: remove 0-NA countries for barplots
# =================================================

def filter_no_na(df_half):
    na_counts = df_half.set_index("Country").isna().sum(axis=1)
    keep_countries = na_counts[na_counts > 0].index
    return df_half[df_half["Country"].isin(keep_countries)]


# =================================================
# 3. Rename columns
# =================================================

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

df = df.rename(columns={c: rename_dict.get(c, c) for c in df.columns})


# =================================================
# 4. Split dataset in half + replace country names with numbers
# =================================================

midpoint = len(df) // 2

df_first_half = df.iloc[:midpoint].copy()
df_second_half = df.iloc[midpoint:].copy()

df_first_half["Country"] = range(1, len(df_first_half) + 1)
df_second_half["Country"] = range(1, len(df_second_half) + 1)


# =================================================
# 5. Plot NA per country (FILTERING 0-NA)
# =================================================

def plot_na_per_country(df_half, title):

    df_filtered = filter_no_na(df_half)

    if df_filtered.empty:
        print(f"No missing values in {title}.")
        return

    na_counts = df_filtered.set_index("Country").isna().sum(axis=1).sort_values(ascending=False)

    plt.figure(figsize=(14, 6))
    na_counts.plot(kind="bar", color="firebrick")
    plt.title(title)
    plt.ylabel("Number of NA")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


# =================================================
# 6. Analyze one half of the dataset
# =================================================

def analyze_half(df_half, half_name):

    print(f"\n=== Analyzing {half_name} — {len(df_half)} countries ===")

    # Barplot with filtered countries
    plot_na_per_country(df_half, f"NA per country – {half_name}")

    # NA per variable
    na_by_var = df_half.isna().sum().sort_values(ascending=False)

    plt.figure(figsize=(20, 6))
    na_by_var.plot(kind="bar", color="royalblue")
    plt.title(f"NA per variable – {half_name}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    # Column groups
    groups = {
        "Footprint Production": [
            col for col in df_half.columns if "_Footprint_Production" in col or col == "Total_Footprint_Production"
        ],
        "Footprint Consumption": [
            col for col in df_half.columns if "_Footprint_Consumption" in col or col == "Total_Footprint_Consumption"
        ],
        "Biocapacity": [
            col for col in df_half.columns if col.endswith("_Biocapacity") or col == "Total_Biocapacity"
        ],
        "Socio-Economic Indicators": [
            col for col in df_half.columns if col in [
                "Quality Score", "SDGi", "Life Expectancy", "HDI", "Per Capita GDP",
                "Region", "Income Group", "Population (millions)",
                "Ecological (Deficit) or Reserve", "Number of Earths required",
                "Number of Countries required", "Overshoot Day"
            ]
        ]
    }

    groups = {g: cols for g, cols in groups.items() if len(cols) > 0}

    # Heatmap function (unchanged)
    def heatmap_na(df_block, title, color="#d62728"):
        sorted_cols = df_block.isna().sum().sort_values(ascending=False).index
        df_sorted = df_block[sorted_cols]

        plt.figure(figsize=(18, 10))
        sns.heatmap(
            df_sorted.isna(),
            cmap=["white", color],
            cbar=False,
            linewidths=0.3,
            linecolor="lightgray"
        )
        plt.title(title, fontsize=16)
        plt.xlabel("Variables")
        plt.ylabel("Countries")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.show()

    # Generate grouped heatmaps
    for group_name, columns in groups.items():
        block = df_half[["Country"] + columns].set_index("Country")
        color = "#8b0000" if group_name == "Socio-Economic Indicators" else "#d62728"
        heatmap_na(block, f"{group_name} – {half_name}", color)


# =================================================
# 7. Run analysis on both halves
# =================================================

analyze_half(df_first_half, "FIRST HALF")
analyze_half(df_second_half, "SECOND HALF")
