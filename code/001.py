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
