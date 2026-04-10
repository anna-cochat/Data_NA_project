import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.patches as mpatches

##### Chargement des données #####
data = pd.read_excel("NA.xlsx")

##### Renommage des variables #####
rename_dict = {
    "actual \nCountry Overshoot Day \n2018": "Overshoot_Day",
    "Cropland Footprint":                    "Cropland_Footprint_Production",
    "Grazing Footprint":                     "Grazing_Footprint_Production",
    "Forest Product Footprint":              "Forest_Footprint_Production",
    "Fish Footprint":                        "Fish_Footprint_Production",
    "Built up land":                         "BuiltUp_Footprint_Production",
    "Carbon Footprint":                      "Carbon_Footprint_Production",
    "Cropland Footprint.1":                  "Cropland_Footprint_Consumption",
    "Grazing Footprint.1":                   "Grazing_Footprint_Consumption",
    "Forest Product Footprint.1":            "Forest_Footprint_Consumption",
    "Fish Footprint.1":                      "Fish_Footprint_Consumption",
    "Built up land.1":                       "BuiltUp_Footprint_Consumption",
    "Carbon Footprint.1":                    "Carbon_Footprint_Consumption",
    "Built up land.2":                       "BuiltUp_Biocapacity",
    "Total biocapacity ":                    "Total_Biocapacity",
    "Total Ecological Footprint (Production)":  "Total_Footprint_Production",
    "Total Ecological Footprint (Consumption)": "Total_Footprint_Consumption",
}
data = data.rename(columns=rename_dict)

##### Préparation #####

# Matrice binaire : 1 = NA, 0 = présent
na_matrix = data.isna().astype(int).values

# Couleurs
MANGO_ORANGE  = "#FF7029"
SPRING_GREEN  = "#00FF7F"
RASPBERRY_RED = "#c21047"

# Ticks Y allégés (20 valeurs espacées)
y_ticks = np.linspace(0, len(data) - 1, 20, dtype=int)

##### Figure 1 – Heatmap des valeurs manquantes #####

cmap1 = ListedColormap(["white", RASPBERRY_RED])

fig, ax = plt.subplots(figsize=(18, 10))

ax.imshow(na_matrix, cmap=cmap1, aspect="auto")

ax.set_xticks(np.arange(len(data.columns)))
ax.set_xticklabels(data.columns, rotation=90, fontsize=8)

ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks + 1, fontsize=8)

orange_patch = mpatches.Patch(color=RASPBERRY_RED, label="NA (manquant)")
white_patch  = mpatches.Patch(facecolor="white", edgecolor="black", label="Présent")

ax.set_title("Heatmap des données manquantes (rouge = NA, blanc = présent)", fontsize=14)
ax.set_xlabel("Variables", fontsize=12)

plt.tight_layout()
plt.savefig("fig1_heatmap_na.png", dpi=150, bbox_inches="tight")
plt.show()

##### Figure 2 – Nombre de valeurs manquantes par variable #####

fig, ax = plt.subplots(figsize=(12, 5))

na_counts = data.isna().sum().sort_values(ascending=False)
na_counts.plot(kind="bar", ax=ax, color="#035063", edgecolor="white")

ax.set_title("Nombre de valeurs manquantes par variable", fontsize=14)
ax.set_xlabel("Variable", fontsize=12)
ax.set_ylabel("Nombre de NA", fontsize=12)
ax.tick_params(axis="x", rotation=90, labelsize=8)

plt.tight_layout()
plt.savefig("fig2_na_par_variable.png", dpi=150, bbox_inches="tight")
plt.show()

##### Figure 3 – Matrice de corrélation #####

numeric_cols = data.select_dtypes(include="number").columns
corr_matrix  = data[numeric_cols].corr()

cmap3 = LinearSegmentedColormap.from_list("teal_to_red", ["#035063", "#f5f5f0", "#c21047"])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=False, cmap=cmap3, ax=ax)
ax.set_title("Matrice de corrélation entre variables numériques", fontsize=14)

plt.tight_layout()
plt.savefig("fig3_correlation.png", dpi=150, bbox_inches="tight")
plt.show()