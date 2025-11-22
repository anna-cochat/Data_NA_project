import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("mespays.xlsx")
print("Columns before:", df.columns.tolist(), "\n")

rename_dict = {
    "actual \nCountry Overshoot Day \n2018": "Overshoot Day",
    # Cropland
    "Cropland Footprint": "Cropland_Footprint_Production",
    "Cropland Footprint.1": "Cropland_Footprint_Consumption",
    "Cropland": "Cropland_Biocapacity",

    # Grazing
    "Grazing Footprint": "Grazing_Footprint_Production",
    "Grazing Footprint.1": "Grazing_Footprint_Consumption",
    "Grazing land": "Grazing_Biocapacity",

    # Forest
    "Forest Product Footprint": "Forest_Footprint_Production",
    "Forest Product Footprint.1": "Forest_Footprint_Consumption",
    "Forest land": "Forest_Biocapacity",

    # Fishing
    "Fish Footprint": "Fish_Footprint_Production",
    "Fish Footprint.1": "Fish_Footprint_Consumption",
    "Fishing ground": "Fishing_Biocapacity",

    # Built up land
    "Built up land": "BuiltUp_Footprint_Production",
    "Built up land.1": "BuiltUp_Footprint_Consumption",
    "Built up land.2": "BuiltUp_Biocapacity",

    # Carbon
    "Carbon Footprint": "Carbon_Footprint_Production",
    "Carbon Footprint.1": "Carbon_Footprint_Consumption",

    # Totals
    "Total Ecological Footprint (Production)": "Total_Footprint_Production",
    "Total Ecological Footprint (Consumption)": "Total_Footprint_Consumption",
    "Total biocapacity": "Total_Biocapacity"
}

df = df.rename(columns={c: rename_dict[c] for c in df.columns if c in rename_dict})

print("Columns after:", df.columns.tolist(), "\n")


force_text = {
    "Country",
    "Region",
    "Income Group",
    "Overshoot Day",
    "Quality Score"
}

def clean_numeric_series(s):
    if s.dtype != object:
        return s

    
    st = s.astype(str).str.strip()

    
    if st.str.contains(r"[A-Za-z]").any():
        return s 

    
    if not st.str.contains(r"\d").any():
        return s

    
    st = (
        st.str.replace("\u202f", "", regex=False)  
          .str.replace(" ", "", regex=False)       
          .str.replace(",", ".", regex=False)      
          .str.replace("$", "", regex=False)
          .str.replace("−", "-", regex=False)      
          .str.replace("%", "", regex=False)
          .str.replace("--", "", regex=False)
          .str.replace("…", "", regex=False)
          .str.replace("\t", "", regex=False)
    )

    
    return pd.to_numeric(st, errors="coerce")


for col in df.columns:
    if col not in force_text:
        df[col] = clean_numeric_series(df[col])


my_countries = [
    "Afghanistan", "Albania", "Algeria", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
    "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus",
    "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei Darussalam",
    "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
    "Cameroon", "Canada", "Central African Republic", "Chad",
    "Chile", "China", "Colombia"
]

mydf = df[df["Country"].isin(my_countries)].copy()


na_counts = mydf.set_index("Country").isna().sum(axis=1).sort_values(ascending=False)

plt.figure(figsize=(12,7))
na_counts.plot(kind="bar", color="firebrick")
plt.title("Nombre de valeurs manquantes par pays")
plt.ylabel("Nombre de NA")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

na_by_var = mydf.isna().sum().sort_values(ascending=False)

plt.figure(figsize=(18,7))
na_by_var.plot(kind="bar", color="royalblue")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.title("Nombre de NA par variable")
plt.tight_layout()
plt.show()


groups = {
    "Footprint Production": [
        col for col in mydf.columns
        if "_Footprint_Production" in col or col == "Total_Footprint_Production"
    ],
    "Footprint Consumption": [
        col for col in mydf.columns
        if "_Footprint_Consumption" in col or col == "Total_Footprint_Consumption"
    ],
    "Biocapacity": [
        col for col in mydf.columns
        if col.endswith("_Biocapacity") or col == "Total_Biocapacity"
    ],
    "Socio-Economic Indicators": [
        col for col in mydf.columns
        if col in [
            "Quality Score", "SDGi", "Life Expectancy", "HDI", "Per Capita GDP",
            "Region", "Income Group", "Population (millions)",
            "Ecological (Deficit) or Reserve", "Number of Earths required",
            "Number of Countries required", "Overshoot_Day_2018"
        ]
    ]
}

for key in list(groups.keys()):
    groups[key] = [col for col in groups[key] if col in mydf.columns]
    if not groups[key]:
        del groups[key]


def heatmap_na(df, title, missing_color="#d62728"):
    sorted_cols = df.isna().sum().sort_values(ascending=False).index
    df_sorted = df[sorted_cols]

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        df_sorted.isna(),
        cmap=["white", missing_color],
        cbar=False,
        linewidths=0.2,
        linecolor="lightgray"
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Variables")
    plt.ylabel("Countries")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.show()

for group_name, columns in groups.items():
    df_block = mydf[["Country"] + columns].set_index("Country")

    if group_name == "Socio-Economic Indicators":
        color = "#8b0000"     
    else:
        color = "#d62728"    

    heatmap_na(df_block, f"Heatmap des valeurs manquantes – {group_name}", missing_color=color)
    
    
def safe(df, cols):
    return df.dropna(subset=cols)
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


mydf["NA_count"] = mydf.isna().sum(axis=1)

threshold = max(5, mydf["NA_count"].quantile(0.90))
outliers = mydf[mydf["NA_count"] >= threshold]

print("=== OUTLIERS DETECTED ===")
print(outliers[["Country", "HDI", "Per Capita GDP", "NA_count"]])
print("\nThreshold used:", threshold)


plt.figure(figsize=(8,6))
sns.scatterplot(data=mydf, x="HDI", y="NA_count", color="firebrick")

for _, row in outliers.iterrows():
    plt.annotate(
        f"{row['Country']}\nHDI={row['HDI']:.2f}\nNA={row['NA_count']}",
        (row["HDI"], row["NA_count"]),
        textcoords="offset points",
        xytext=(5,5),
        fontsize=7
    )

plt.title("Relation entre HDI et nombre de valeurs manquantes")
plt.xlabel("HDI")
plt.ylabel("Nombre de NA")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(data=mydf, x="Per Capita GDP", y="NA_count", color="teal")

for _, row in outliers.iterrows():
    plt.annotate(
        f"{row['Country']}\nGDP={row['Per Capita GDP']:.0f}\nNA={row['NA_count']}",
        (row["Per Capita GDP"], row["NA_count"]),
        textcoords="offset points",
        xytext=(5,5),
        fontsize=7
    )

plt.title("Relation entre PIB par habitant et nombre de NA")
plt.xlabel("PIB par habitant")
plt.ylabel("Nombre de NA")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.boxplot(data=mydf, x="Income Group", y="NA_count")
sns.stripplot(data=mydf, x="Income Group", y="NA_count", color="black", size=3)

plt.title("Distribution des NA selon le niveau de revenu")
plt.xlabel("Groupe de revenu")
plt.ylabel("Nombre de NA")
plt.tight_layout()
plt.show()

num_cols = ["HDI", "Per Capita GDP", "Life Expectancy", "NA_count"]
num_cols = [c for c in num_cols if c in mydf.columns]

corr = mydf[num_cols].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélations (incluant le nombre de NA)")
plt.tight_layout()
plt.show()



df_angle2 = mydf.dropna(subset=["Total_Footprint_Consumption", "Total_Biocapacity"]).copy()

df_angle2["Eco_Balance"] = df_angle2["Total_Biocapacity"] - df_angle2["Total_Footprint_Consumption"]

Q1 = df_angle2["Eco_Balance"].quantile(0.25)
Q3 = df_angle2["Eco_Balance"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df_angle2[(df_angle2["Eco_Balance"] < lower) | (df_angle2["Eco_Balance"] > upper)]

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_angle2,
    x="Total_Footprint_Consumption",
    y="Total_Biocapacity",
    s=df_angle2["Population (millions)"] * 5, 
    color="darkred",
    alpha=0.7
)

plt.xlabel("Empreinte écologique (consommation)")
plt.ylabel("Biocapacité totale")
plt.title("Relation entre Empreinte de consommation et Biocapacité")

max_val = max(df_angle2["Total_Footprint_Consumption"].max(),
              df_angle2["Total_Biocapacity"].max())
plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--", label="Équilibre")

for _, row in outliers.iterrows():
    plt.annotate(
        f"{row['Country']}",
        (row["Total_Footprint_Consumption"], row["Total_Biocapacity"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8
    )

plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_angle2.sort_values("Eco_Balance"),
    x="Country",
    y="Eco_Balance",
    palette=["darkred" if x < 0 else "forestgreen" for x in df_angle2.sort_values("Eco_Balance")["Eco_Balance"]]
)

plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Biocapacité – Empreinte (équilibre positif)")
plt.title("Différence entre Biocapacité et Empreinte écologique")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df_angle2["Eco_Balance"], bins=10, kde=True, color="steelblue")
plt.axvline(0, color="black", linestyle="--")
plt.title("Distribution des déficits / réserves écologiques")
plt.xlabel("Biocapacité – Empreinte")
plt.ylabel("Nombre de pays")
plt.tight_layout()
plt.show()


print("\n=== OUTLIERS — Écarts Extrêmes ===")
print(outliers[["Country", "Total_Footprint_Consumption", "Total_Biocapacity", "Eco_Balance"]])
print("\nSeuils utilisés (IQR):")
print(f"Borne basse = {lower:.2f}")
print(f"Borne haute = {upper:.2f}")
