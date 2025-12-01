# Pays de Afghanistan à Colombie
import pandas as pd
import matplotlib.pyplot as plt

def load_mydf():  # On wrap le tout dans une fonction à réutiliser 

    df = pd.read_excel("data/mespays.xlsx") # Charger le fichier Excel
    print("Columns before:", df.columns.tolist(), "\n") # Afficher les colonnes avant le renommage

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
    } # Dictionnaire de renommage des colonnes

    df = df.rename(columns={c: rename_dict[c] for c in df.columns if c in rename_dict}) # Renommer les colonnes

    print("Columns after:", df.columns.tolist(), "\n") # Afficher les colonnes après le renommage


    force_text = {
        "Country",
        "Region",
        "Income Group",
        "Overshoot Day",
        "Quality Score"
    } # Colonnes à forcer en texte

    def clean_numeric_series(s): # Fonction pour nettoyer les var numériques
        if s.dtype != object: # Si ce n'est pas du texte
            return s # Retourner tel quel

        st = s.astype(str).str.strip() # Convertir en texte et enlever espaces

        if st.str.contains(r"[A-Za-z]").any(): # Si des lettres sont présentes
            return s

        if not st.str.contains(r"\d").any(): # Si pas de chiffres
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
        ) # Nettoyer le texte

        return pd.to_numeric(st, errors="coerce") # Convertir en numérique, NA si erreur

    for col in df.columns:
        if col not in force_text:
            df[col] = clean_numeric_series(df[col]) # Nettoyer les colonnes non texte

    my_countries = [
        "Afghanistan", "Albania", "Algeria", "Angola", "Antigua and Barbuda",
        "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
        "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus",
        "Belgium", "Belize", "Benin", "Bhutan", "Bolivia",
        "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei Darussalam",
        "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia",
        "Cameroon", "Canada", "Central African Republic", "Chad",
        "Chile", "China", "Colombia"
    ] # Liste des pays à analyser

    mydf = df[df["Country"].isin(my_countries)].copy() # Filtrer les pays

    return mydf 


from preprocessing import load_mydf
import seaborn as sns
import matplotlib.pyplot as plt
import os

mydf = load_mydf() 

FIG_DIR = "figures/001"

def savefig(name):
    """Sauvegarde automatique des figures dans figures/001"""
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")


na_counts = mydf.set_index("Country").isna().sum(axis=1).sort_values(ascending=False)
#  compte le nombre de NA pour chaque pays, puis trie du + NA au - NA

plt.figure(figsize=(12,7))
na_counts.plot(kind="bar", color="firebrick")
plt.title("Nombre de valeurs manquantes par pays")
plt.ylabel("Nombre de NA")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
savefig("001_NAparpays")
plt.show()

na_by_var = mydf.isna().sum().sort_values(ascending=False)
# -> compte les NA variable par variable

plt.figure(figsize=(18,7))
na_by_var.plot(kind="bar", color="royalblue")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.title("Nombre de NA par variable")
plt.tight_layout()
savefig("001_NAparvariable")
plt.show()


# Définition des groupes thématiques pour comprendre quelles familles de variables manquent
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

# Nettoyage des groupes (c'était pour fix un bug)
for key in list(groups.keys()):
    groups[key] = [col for col in groups[key] if col in mydf.columns]
    if not groups[key]:
        del groups[key]


# Heatmap des na
def heatmap_na(df, title, missing_color="#d62728"):

    sorted_cols = df.isna().sum().sort_values(ascending=False).index
    # trie les colonnes dans le heatmap selon le nombre de NA

    df_sorted = df[sorted_cols]

    plt.figure(figsize=(14, 10))
    sns.heatmap(
        df_sorted.isna(),
        cmap=["white", missing_color],    # blanc = ok, rouge = NA
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


# une heatmap pour chaque thématique
for group_name, columns in groups.items():

    df_block = mydf[["Country"] + columns].set_index("Country")

    # Rouge foncé pour socio-éco, rouge normal pour le reste
    if group_name == "Socio-Economic Indicators":
        color = "#8b0000"
    else:
        color = "#d62728"

    heatmap_na(df_block, f"Heatmap des valeurs manquantes – {group_name}", missing_color=color)
    savefig(f"001_heatmap_{group_name.replace(' ','_')}")
    plt.show()


# Outliers

mydf["NA_count"] = mydf.isna().sum(axis=1)
# ajoute une colonne NA_count = nombre total de valeurs manquantes

threshold = max(5, mydf["NA_count"].quantile(0.90))
#  seuil = max(5 NA, 90e percentile)

outliers = mydf[mydf["NA_count"] >= threshold]

print("Outliers détectés")
print(outliers[["Country", "HDI", "Per Capita GDP", "NA_count"]])
print("\nLimite:", threshold)


# Scatter HDI vs NA_count
plt.figure(figsize=(8,6))
sns.scatterplot(data=mydf, x="HDI", y="NA_count", color="firebrick")

# Annotation des outliers
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
savefig("001_HDIvsNA")
plt.show()

# Scatter PIB vs NA_count
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
savefig("001_PIBvsNA")
plt.show()


# Boxplot NA_count - revenu
plt.figure(figsize=(8,6))
sns.boxplot(data=mydf, x="Income Group", y="NA_count")
sns.stripplot(data=mydf, x="Income Group", y="NA_count", color="black", size=3)

plt.title("Distribution des NA selon le niveau de revenu")
plt.xlabel("Groupe de revenu")
plt.ylabel("Nombre de NA")
plt.tight_layout()
savefig("001_NAparIncome")
plt.show()


# Matrice de corrélation
num_cols = ["HDI", "Per Capita GDP", "Life Expectancy", "NA_count"]
num_cols = [c for c in num_cols if c in mydf.columns]

corr = mydf[num_cols].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Corrélations (incluant le nombre de NA)")
plt.tight_layout()
savefig("001_matricecorrelation")
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import os

# =====================================================
# 0. FIGURE SAVING SYSTEM (your format)
# =====================================================

FIG_DIR = "code"

def savefig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")

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
# 2. BLOCK DEFINITIONS FOR COMPACT MISSINGNESS
# =====================================================

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
    """Compact missingness by block."""
    output = []

    if all(v in vars_list for v in block_prod):
        output.append("Missing ALL Production Footprint components")
        vars_list = [v for v in vars_list if v not in block_prod]

    if all(v in vars_list for v in block_cons):
        output.append("Missing ALL Consumption Footprint components")
        vars_list = [v for v in vars_list if v not in block_cons]

    if all(v in vars_list for v in block_bio):
        output.append("Missing ALL Biocapacity components")
        vars_list = [v for v in vars_list if v not in block_bio]

    if vars_list:
        output.append("Missing individual variables: " + ", ".join(vars_list))

    return output


# =====================================================
# 3. RUN ANALYSIS + GENERATE TXT + SAVE FIGURES
# =====================================================

df = load_mydf()
df["NA_count"] = df.isna().sum(axis=1)

numeric_vars = df.select_dtypes(include="number").columns.tolist()
numeric_vars.remove("NA_count")

threshold = 3

output_path = "missingness_report.txt"
with open(output_path, "w", encoding="utf-8") as f:

    index = 1  # for naming figures 001, 002, etc.

    for var_x in numeric_vars:

        f.write("\n\n" + "="*60 + "\n")
        f.write(f"VARIABLE STUDIED : {var_x}\n")
        f.write("="*60 + "\n\n")

        df_dropped = df[df[var_x].isna()].copy()
        df_highNA = df[df["NA_count"] >= threshold].copy()

        # -------------------------
        # DROPPED COUNTRIES
        # -------------------------
        f.write("--- COUNTRIES DROPPED (X missing) ---\n\n")
        for _, row in df_dropped.sort_values("NA_count", ascending=False).iterrows():
            c = row["Country"]
            missing_vars = row[row.isna()].index.tolist()
            summary = compress_missing(missing_vars)

            f.write(f"{c:25s} | NA={row['NA_count']}\n")
            for line in summary:
                f.write(f"   → {line}\n")
            f.write("\n")

        # -------------------------
        # HIGH NA COUNTRIES
        # -------------------------
        f.write("\n--- COUNTRIES WITH HIGH NA COUNT (>=3) ---\n\n")
        for _, row in df_highNA.sort_values("NA_count", ascending=False).iterrows():
            if pd.isna(row[var_x]):
                continue

            c = row["Country"]
            missing_vars = row[row.isna()].index.tolist()
            summary = compress_missing(missing_vars)

            f.write(f"{c:25s} | {var_x}={row[var_x]} | NA={row['NA_count']}\n")
            for line in summary:
                f.write(f"   → {line}\n")
            f.write("\n")

        # -------------------------
        # SAVE FIGURE
        # -------------------------
        df_used = df[df[var_x].notna()]
        plt.figure(figsize=(10, 6))
        plt.scatter(df_used[var_x], df_used["NA_count"], alpha=0.7)
        plt.xlabel(var_x)
        plt.ylabel("NA_count")
        plt.title(f"NA_count vs {var_x}")
        plt.grid(alpha=0.3)

        fig_name = f"NA_vs_{var_x}"
        savefig(fig_name)
        plt.show()

        index += 1

print("\n📄 TXT generated: missingness_report.txt")




# Angle 3 — Développement

from preprocessing import load_mydf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

mydf = load_mydf()  

FIG_DIR = "figures/003"

def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")

# Préparation de la variable éco balance

df3 = mydf.copy()
df3["Eco_Balance"] = df3["Total_Biocapacity"] - df3["Total_Footprint_Consumption"]

# HDI vs Footprint Consumption

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df3, x="HDI", y="Total_Footprint_Consumption",
    color="darkred", s=70
)

# lowess (regression douce)
low = lowess(df3["Total_Footprint_Consumption"], df3["HDI"], frac=0.6)
plt.plot(low[:,0], low[:,1], color="black", linewidth=2)

plt.title("HDI vs Empreinte de Consommation")
plt.xlabel("HDI")
plt.ylabel("Empreinte écologique (consommation)")

# annoter quelques pays extrêmes
top = df3.sort_values("Total_Footprint_Consumption", ascending=False).head(3)
bottom = df3.sort_values("Total_Footprint_Consumption").head(3)

for _, r in pd.concat([top, bottom]).iterrows():
    plt.annotate(r["Country"], (r["HDI"], r["Total_Footprint_Consumption"]),
        textcoords="offset points", xytext=(4,4), fontsize=7)

savefig("003_HDI_vs_Footprint")
plt.show()


# GDP vs Footprint Consumption

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df3, x="Per Capita GDP", y="Total_Footprint_Consumption",
    color="steelblue", s=70
)

low = lowess(df3["Total_Footprint_Consumption"], df3["Per Capita GDP"], frac=0.6)
plt.plot(low[:,0], low[:,1], color="black", linewidth=2)

plt.title("PIB par habitant vs Empreinte de Consommation")
plt.xlabel("PIB par habitant")
plt.ylabel("Empreinte écologique (consommation)")

for _, r in pd.concat([top, bottom]).iterrows():
    plt.annotate(r["Country"], (r["Per Capita GDP"], r["Total_Footprint_Consumption"]),
        textcoords="offset points", xytext=(4,4), fontsize=7)

savefig("003_GDP_vs_Footprint")
plt.show()


# HDI vs Ecological Deficit

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df3, x="HDI", y="Eco_Balance",
    color="forestgreen", s=70
)

low = lowess(df3["Eco_Balance"], df3["HDI"], frac=0.6)
plt.plot(low[:,0], low[:,1], color="black", linewidth=2)

plt.axhline(0, color="gray", linestyle="--")  # ligne d'équilibre

plt.title("HDI vs Équilibre Écologique (Biocapacité – Empreinte)")
plt.xlabel("HDI")
plt.ylabel("Équilibre écologique")

# annoter pays en déficit extrême
ext = df3.sort_values("Eco_Balance").head(3)
for _, r in ext.iterrows():
    plt.annotate(r["Country"], (r["HDI"], r["Eco_Balance"]),
        textcoords="offset points", xytext=(4,4), fontsize=7)

savefig("003_HDI_vs_EcoBalance")
plt.show()


# On note les pays un peu paradoxaux
#   (haut HDI mais faible empreinte)

paradox = df3[
    (df3["HDI"] >= 0.75) &
    (df3["Total_Footprint_Consumption"] <= 2.5) &
    (df3["Eco_Balance"] >= -1)
][["Country", "HDI", "Total_Footprint_Consumption", "Eco_Balance"]]

print("\n PAYS PARADOXAUX (HDI élevé, faible empreinte) ")
print(paradox)



from preprocessing import load_mydf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from statsmodels.nonparametric.smoothers_lowess import lowess

mydf = load_mydf()  
print(mydf.columns)

FIG_DIR = "figures/004"

def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")


# On calcule l'écart
mydf['Footprint_Gap'] = mydf['Total_Footprint_Consumption'] - mydf['Total_Footprint_Production']

plt.figure(figsize=(16,6))
ordered = mydf.sort_values('Footprint_Gap')

plt.bar(ordered['Country'], ordered['Footprint_Gap'],
        color=['darkgreen' if x >= 0 else 'darkred' for x in ordered['Footprint_Gap']])

plt.xticks(rotation=90)
plt.axhline(0, color='black', linewidth=1)
plt.title("Gap Empreinte : Consommation - Production (positif = externalisation)")
plt.ylabel("Gap (gha/personne)")
plt.tight_layout()

savefig("004_GapBarplot")
plt.show()


# Production vs consommation
plt.figure(figsize=(8,6))
plt.scatter(mydf['Total_Footprint_Production'], mydf['Total_Footprint_Consumption'], s=80)

max_val = max(mydf['Total_Footprint_Production'].max(), mydf['Total_Footprint_Consumption'].max())
plt.plot([0, max_val], [0, max_val], '--', color='gray')

plt.xlabel("Empreinte de Production")
plt.ylabel("Empreinte de Consommation")
plt.title("Production vs Consommation")

plt.tight_layout()
savefig("004_Production_vs_Consumption")
plt.show()


# top 10 externaliseurs
top_ext = mydf.sort_values('Footprint_Gap', ascending=False).head(10)

plt.figure(figsize=(10,5))
plt.barh(top_ext['Country'], top_ext['Footprint_Gap'], color='darkgreen')
plt.gca().invert_yaxis()

plt.xlabel("Gap (Consommation - Production)")
plt.title("Top externalisateurs d'empreinte écologique")
plt.tight_layout()

savefig("004_TopExternalizers")
plt.show()




from preprocessing import load_mydf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

mydf = load_mydf()

FIG_DIR = "figures/005"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    """Save figures to the correct folder."""
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")


plt.figure(figsize=(10,6))
sns.boxplot(data=mydf, x="Region", y="Total_Footprint_Consumption")
plt.xticks(rotation=45)
plt.ylabel("Empreinte (consommation)")
plt.title("Empreinte écologique de consommation par région")
plt.tight_layout()
savefig("005_Box_Footprint_Region")
plt.show()


plt.figure(figsize=(10,6)) 
sns.boxplot(data=mydf, x="Region", y="Total_Biocapacity")
plt.xticks(rotation=45)
plt.ylabel("Biocapacité totale")
plt.title("Biocapacité par région")
plt.tight_layout()
savefig("005_Box_Biocapacity_Region")
plt.show()

mydf["Eco_Balance"] = mydf["Total_Biocapacity"] - mydf["Total_Footprint_Consumption"]

plt.figure(figsize=(10,6))
sns.boxplot(data=mydf, x="Region", y="Eco_Balance")
plt.xticks(rotation=45)
plt.ylabel("Biocapacité – Empreinte")
plt.title("Équilibre écologique par région")
plt.tight_layout()
savefig("005_Box_Ecobal_Region")
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=mydf, x="Region", y="Total_Footprint_Consumption", showfliers=False)
sns.stripplot(data=mydf, x="Region", y="Total_Footprint_Consumption",
              color="black", alpha=0.6, jitter=True)
plt.xticks(rotation=45)
plt.ylabel("Empreinte écologique (consommation)")
plt.title("Dispersion des empreintes écologiques par région")
plt.tight_layout()
savefig("005_Strip_Footprint_Region")
plt.show()


regional_vars = [
    "Total_Footprint_Consumption",
    "Total_Footprint_Production",
    "Total_Biocapacity",
    "Eco_Balance",
    "HDI",
    "Per Capita GDP"
]

regional_mean = mydf.groupby("Region")[regional_vars].mean().round(2)

plt.figure(figsize=(10,6))
sns.heatmap(regional_mean, annot=True, cmap="YlGnBu", linewidths=0.5)
plt.title("Moyennes régionales des indicateurs clés")
plt.tight_layout()
savefig("005_Heatmap_Regional_Averages")
plt.show()


plt.figure(figsize=(10,5))
sns.barplot(
    data=regional_mean.reset_index(),
    x="Region",
    y="Total_Footprint_Consumption",
    color="darkred"
)
plt.xticks(rotation=45)
plt.ylabel("Empreinte moyenne (consommation)")
plt.title("Empreinte écologique moyenne par région")
plt.tight_layout()
savefig("005_Regional_Average_Footprint")
plt.show



from preprocessing import load_mydf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

mydf = load_mydf()

FIG_DIR = "figures/006"

def savefig(name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}")

mydf["Pop_Quartile"] = pd.qcut(mydf["Population (millions)"], 4, labels=["Q1","Q2","Q3","Q4"])

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=mydf,
    x="Per Capita GDP",
    y="Total_Footprint_Consumption",
    hue="Pop_Quartile",
    palette="viridis",
    s=90
)

plt.title("Empreinte écologique vs PIB par habitant")
plt.xlabel("PIB par habitant")
plt.ylabel("Empreinte écologique (consommation)")
plt.legend(title="Population quartile")

savefig("006_Footprint_vs_GDP")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=mydf,
    x="HDI",
    y="Total_Biocapacity",
    hue="Pop_Quartile",
    palette="viridis",
    s=90
)

plt.title("Biocapacité vs HDI")
plt.xlabel("HDI")
plt.ylabel("Biocapacité totale")
plt.legend(title="Population quartile")

savefig("006_Biocapacity_vs_HDI")
plt.show()

plt.figure(figsize=(9,7))
sns.scatterplot(
    data=mydf,
    x="Total_Biocapacity",
    y="Total_Footprint_Consumption",
    hue="Pop_Quartile",
    palette="viridis",
    s=120
)


max_val = max(
    mydf["Total_Biocapacity"].max(),
    mydf["Total_Footprint_Consumption"].max()
)
plt.plot([0, max_val], [0, max_val], '--', color='gray')

plt.title("Empreinte vs Biocapacité (couleurs = quartiles de population)")
plt.xlabel("Biocapacité")
plt.ylabel("Empreinte écologique")

savefig("006_Inequality_PopQuartiles")
plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from preprocessing import load_mydf

# Load data
df = load_mydf()

# Create output folder
FIG_DIR = "figures/007"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")

# -------------------------------------------------------
# 1. Compute Missingness per Country
# -------------------------------------------------------

# % of missing values per row
df["Missingness"] = df.isna().mean(axis=1) * 100

# For heatmaps: missingness per column grouped by region
missing_by_region = df.groupby("Region").apply(lambda g: g.isna().mean() * 100)


# -------------------------------------------------------
# 2. Missingness vs SDGi
# -------------------------------------------------------
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="SDGi",
    y="Missingness",
    hue="Region",
    palette="tab10",
    s=120,
    edgecolor="black"
)

plt.xlabel("SDGi (indice de gouvernance)")
plt.ylabel("% de données manquantes")
plt.title("Taux de données manquantes vs SDGi (par région)")

# Force legend to the side, title added
plt.legend(title="Région", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
savefig("007_Missingness_vs_IncomeGroup")
plt.show()


# -------------------------------------------------------
# 3. Missingness vs Income Group
# -------------------------------------------------------
plt.figure(figsize=(8,6))
sns.boxplot(
    data=df,
    x="Income Group",
    y="Missingness",
    palette="Set2"
)
sns.stripplot(
    data=df,
    x="Income Group",
    y="Missingness",
    color="black",
    size=4,
    alpha=0.7
)
plt.xticks(rotation=30)
plt.ylabel("% de valeurs manquantes")
plt.title("Missingness vs Niveau de revenu")
savefig("007_Missingness_vs_IncomeGroup")
plt.show()

# -------------------------------------------------------
# 4. Heatmap of Missingness by Region
# -------------------------------------------------------

# Remove non-informative columns (identifiers)
cols_to_exclude = ["Country", "Region", "Income Group"]
heatmap_df = missing_by_region.drop(columns=[c for c in cols_to_exclude if c in missing_by_region.columns])

plt.figure(figsize=(14,7))
sns.heatmap(
    heatmap_df,
    cmap="YlOrRd",
    annot=True,
    fmt=".1f",
    cbar_kws={"label": "% de valeurs manquantes"}
)
plt.title("Heatmap des valeurs manquantes par région")
plt.ylabel("Région")
plt.xlabel("Variables")
savefig("007_Heatmap_Region_Missingness")
plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi
from preprocessing import load_mydf

# Load data
df = load_mydf()

FIG_DIR = "figures/008"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")

# === 1) Select footprint components (Consumption version) ===

components = [
    "Cropland_Footprint_Consumption",
    "Grazing_Footprint_Consumption",
    "Forest_Footprint_Consumption",
    "Fish_Footprint_Consumption",
    "BuiltUp_Footprint_Consumption",
    "Carbon_Footprint_Consumption"
]

df_comp = df[["Country"] + components].set_index("Country")
df_comp = df_comp.applymap(
    lambda x: float(str(x).replace(",", ".")) if isinstance(x, str) else x
)
# =====================================================================
# 1. HEATMAP — Distribution of footprint components
# =====================================================================

plt.figure(figsize=(12,7))
sns.heatmap(df_comp, cmap="viridis", annot=False)
plt.title("Composition de l'empreinte écologique (Consommation) — Heatmap")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/009_Heatmap_Composition.png", dpi=300)
plt.show()


# =====================================================================
# 2. STACKED BARPLOT — Composition by country
# =====================================================================

df_stacked = df_comp.copy()

plt.figure(figsize=(14,8))
df_stacked.plot(kind="bar", stacked=True, colormap="tab20", figsize=(14,8))

plt.title("Empreinte écologique — contribution des composantes (stacked)")
plt.ylabel("gha/personne")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/009_StackedBar_Composition.png", dpi=300)
plt.show()



from preprocessing import load_mydf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

FIG_DIR = "figures/009"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {name}")

df = load_mydf()

# -------- 1) Variables for clustering --------
features = [
    "Total_Footprint_Consumption",
    "Total_Footprint_Production",
    "Total_Biocapacity",
    "Ecological (Deficit) or Reserve",
    "HDI",
    "Per Capita GDP",
    "Population (millions)"
]

data = df[features].copy()

# Handle NaN
data = data.fillna(data.mean())

# -------- 2) Scaling --------
scaler = StandardScaler()
X = scaler.fit_transform(data)

# -------- 3) PCA --------
pca = PCA(n_components=2)
pca_coords = pca.fit_transform(X)

df["PC1"] = pca_coords[:, 0]
df["PC2"] = pca_coords[:, 1]

plt.figure(figsize=(9,7))
sns.scatterplot(x="PC1", y="PC2", data=df, hue="Region", s=80, palette="tab10")
for i in range(len(df)):
    plt.text(df["PC1"][i]+0.05, df["PC2"][i], df["Country"][i], fontsize=7)

plt.title("PCA — Clustering écologique (2D)")
plt.tight_layout()
savefig("010_PCA_2D")
plt.show()

# -------- 4) KMeans clustering --------
kmeans = KMeans(n_clusters=4, random_state=42)
df["Cluster"] = kmeans.fit_predict(X)

plt.figure(figsize=(9,7))
sns.scatterplot(x="PC1", y="PC2", data=df, hue="Cluster", palette="Set2", s=90)
for i in range(len(df)):
    plt.text(df["PC1"][i]+0.05, df["PC2"][i], df["Country"][i], fontsize=7)
plt.title("Clustering KMeans (4 groupes)")
plt.tight_layout()
savefig("010_KMeans")
plt.show()

# -------- 5) Dendrogram --------
plt.figure(figsize=(14,6))
linked = linkage(X, method="ward")
dendrogram(linked, labels=df["Country"].values, leaf_rotation=90)
plt.title("Dendrogramme — Clustering hiérarchique")
plt.tight_layout()
savefig("010_Dendrogram")
plt.show()

# -------- 6) Cluster heatmap --------
df_clustered = df.sort_values("Cluster")
heat_vars = [
    "Total_Footprint_Consumption", "Total_Footprint_Production",
    "Total_Biocapacity", "Ecological (Deficit) or Reserve",
    "HDI", "Per Capita GDP"
]

plt.figure(figsize=(12,12))
sns.heatmap(
    df_clustered[heat_vars],
    cmap="vlag",
    annot=False,
    yticklabels=df_clustered["Country"]
)
plt.title("Heatmap — Profils écologiques par cluster")
plt.tight_layout()
savefig("010_Heatmap_Clusters")
plt.show()



from preprocessing import load_mydf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

df = load_mydf().copy()

# ==============================
# 1) Sélection des colonnes numériques
# ==============================
num_cols = df.select_dtypes(include=[float, int]).columns
num_cols = [c for c in num_cols if c not in ["Population (millions)"]]  # optionnel

# ==============================
# 2) Fonctions de détection
# ==============================

def zscore_outliers(x, thresh=3):
    z = (x - x.mean()) / x.std()
    return x.index[z.abs() > thresh].tolist()

def iqr_outliers(x):
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return x.index[(x < lower) | (x > upper)].tolist()

def iso_outliers(x):
    # isolation forest needs 2D
    model = IsolationForest(contamination=0.1, random_state=42)
    preds = model.fit_predict(x.values.reshape(-1, 1))
    return x.index[preds == -1].tolist()

# ==============================
# 3) Tableau final des outliers
# ==============================

results = []

for col in num_cols:
    series = df[col].dropna()

    z_out = zscore_outliers(series)
    iqr_out = iqr_outliers(series)
    iso_out = iso_outliers(series)

    # merge unique outliers
    all_outliers = set(z_out + iqr_out + iso_out)

    for country in all_outliers:
        results.append({
            "Variable": col,
            "Country": df.loc[country, "Country"],
            "Z-score": country in z_out,
            "IQR": country in iqr_out,
            "IsolationForest": country in iso_out
        })

outlier_df = pd.DataFrame(results)

# ==============================
# 4) Export ou affichage
# ==============================
print("\n=== OUTLIERS PAR VARIABLE ===\n")
print(outlier_df.sort_values(["Variable", "Country"]))

# Optionnel : sauvegarde Excel
# outlier_df.to_excel("outliers_par_variable.xlsx", index=False)
# On suppose que outlier_df existe déjà
# Si besoin : outlier_df = pd.read_excel("outliers_par_variable.xlsx")

# 1) Compter combien de fois chaque pays apparaît comme outlier
counts = outlier_df["Country"].value_counts()

# 2) Filtrer ceux qui apparaissent 3 fois ou plus
strong_outliers = counts[counts >= 3]

print("\n=== PAYS QUI APPARAISSENT DANS >= 3 OUTLIERS ===\n")
print(strong_outliers)
