## HEADER — à mettre en tête du fichier
import missingno as msno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns
import geopandas as gpd
import requests
import io
import zipfile

from nettoyage import prepare_data
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # <-- THIS gives LaTeX look
})
# ─────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────
NOMS_PAYS = {
    "Qatar": "Qatar",
    "Luxembourg": "Luxembourg",
    "United Arab Emirates": "Émirats arabes unis",
    "Bahrain": "Bahreïn",
    "Kuwait": "Koweït",
    "Estonia": "Estonie",
    "Belize": "Belize",
    "Trinidad and Tobago": "Trinité-et-Tobago",
    "United States of America": "États-Unis",
    "Australia": "Australie",
    "Belgium": "Belgique",
    "Finland": "Finlande",
    "Latvia": "Lettonie",
    "Sweden": "Suède",
    "Lithuania": "Lituanie",
    "Mongolia": "Mongolie",
    "Korea, Republic of": "Corée du Sud",
    "Denmark": "Danemark",
    "Netherlands": "Pays-Bas",
    "Norway": "Norvège",
    "Iceland": "Islande",
    "Slovenia": "Slovénie",
    "Turkmenistan": "Turkménistan",
    "Saudi Arabia": "Arabie saoudite",
    "Israel": "Israël",
    "Slovakia": "Slovaquie",
    "Germany": "Allemagne",
    "Portugal": "Portugal",
    "Spain": "Espagne",
    "Italy": "Italie",
    "Japan": "Japon",
    "Belarus": "Biélorussie",
    "Switzerland": "Suisse",
    "Chile": "Chili",
    "United Kingdom": "Royaume-Uni",
    "Montenegro": "Monténégro",
    "Barbados": "Barbade",
    "Hungary": "Hongrie",
    "Bulgaria": "Bulgarie",
    "Turkey": "Turquie",
    "Guyana": "Guyana",
    "Bosnia and Herzegovina": "Bosnie-Herzégovine",
    "Panama": "Panama",
    "Republic of North Macedonia": "Macédoine du Nord",
    "Suriname": "Suriname",
    "Serbia": "Serbie",
    "Fiji": "Fidji",
    "Paraguay": "Paraguay",
    "Namibia": "Namibie",
    "Brazil": "Brésil",
    "Mauritania": "Mauritanie",
    "Ukraine": "Ukraine",
    "Costa Rica": "Costa Rica",
    "Botswana": "Botswana",
    "Thailand": "Thaïlande",
    "Peru": "Pérou",
    "Georgia": "Géorgie",
    "Viet Nam": "Viêt Nam",
    "Azerbaijan": "Azerbaïdjan",
    "El Salvador": "Salvador",
    "Albania": "Albanie",
    "Gabon": "Gabon",
    "Guinea": "Guinée",
    "Republic of Moldova": "Moldavie",
    "Morocco": "Maroc",
    "Jordan": "Jordanie",
    "Colombia": "Colombie",
    "Indonesia": "Indonésie",
    "Iraq": "Irak",
    "Cuba": "Cuba",
    "Lao People's Democratic Republic": "Laos",
    "Dominican Republic": "Rép. dominicaine",
    "Papua New Guinea": "Papouasie-N.-Guinée",
    "Uruguay": "Uruguay",
    "Kyrgyzstan": "Kirghizstan",
    "Philippines": "Philippines",
    "Congo": "Congo",
    "Chad": "Tchad",
    "Sri Lanka": "Sri Lanka",
    "India": "Inde",
    "Angola": "Angola",
    "Lesotho": "Lesotho",
    "Senegal": "Sénégal",
    "Kenya": "Kenya",
    "Nigeria": "Nigéria",
    "Mozambique": "Mozambique",
    "Mexico": "Mexique",
    "Yemen": "Yémen",
    "Afghanistan": "Afghanistan",
    "Nepal": "Népal",
    "Sudan": "Soudan",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Tanzania, United Republic of": "Tanzanie",
    "Congo, Democratic Republic of": "Rép. dém. du Congo",
    "Central African Republic": "Rép. centrafricaine",
    "Ethiopia": "Éthiopie",
    "Malawi": "Malawi",
    "Sierra Leone": "Sierra Leone",
    "Uganda": "Ouganda",
    "Rwanda": "Rwanda",
    "Burundi": "Burundi",
    "Haiti": "Haïti",
    "Togo": "Togo",
    "Mali": "Mali",
    "Niger": "Niger",
    "Benin": "Bénin",
}
NOMS_VARIABLES = {
    "SDGi":                           "Indice SDG",
    "Life Expectancy":                "Espérance de vie",
    "HDI":                            "IDH",
    "Per Capita GDP":                 "PIB par habitant",
    "Population (millions)":          "Population (millions)",
    "Cropland_Footprint_Production":  "Empreinte cultures (prod.)",
    "Grazing_Footprint_Production":   "Empreinte pâturages (prod.)",
    "Forest_Footprint_Production":    "Empreinte forêts (prod.)",
    "Fish_Footprint_Production":      "Empreinte pêche (prod.)",
    "BuiltUp_Footprint_Production":   "Empreinte urbain (prod.)",
    "Carbon_Footprint_Production":    "Empreinte carbone (prod.)",
    "Total_Footprint_Production":     "Empreinte totale (prod.)",
    "Cropland_Footprint_Consumption": "Empreinte cultures (conso.)",
    "Grazing_Footprint_Consumption":  "Empreinte pâturages (conso.)",
    "Forest_Footprint_Consumption":   "Empreinte forêts (conso.)",
    "Fish_Footprint_Consumption":     "Empreinte pêche (conso.)",
    "Carbon_Footprint_Consumption":   "Empreinte carbone (conso.)",
    "Total_Footprint_Consumption":    "Empreinte totale (conso.)",
    "Grazing land":                   "Biocapacité pâturages",
    "Forest land":                    "Biocapacité forêts",
    "Fishing ground":                 "Biocapacité pêche",
    "Total_Biocapacity":              "Biocapacité totale",
    "Income Group":                   "Groupe de revenu",
    "Quality Score":                  "Score qualité données",
    "Region":                         "Région",
    "Region_EU":          "Région : Europe",
"Quality Score_2A":   "Score qualité : 2A",
"Income Group_HI":    "Revenu : élevé",
"Income Group_UM":    "Revenu : intermédiaire sup.",
"Region_North America":  "Région : Amérique du Nord",
"Region_South America":  "Région : Amérique du Sud",
"Quality Score_2B":   "Score qualité : 2B",
"Quality Score_3A":   "Score qualité : 3A"
}

VERT_FONCE   = "#2D6A4F"
VERT_MOYEN   = "#52B788"
VERT_CLAIR   = "#95D5B2"
VERT_PALE    = "#D8F3DC"

MAUVE_FONCE  = "#6B3FA0"
MAUVE_MOYEN  = "#9B72CF"
MAUVE_CLAIR  = "#C9B1E8"
MAUVE_PALE   = "#EDE7F6"

GRIS_FONCE   = "#3D3D3D"
GRIS_CLAIR   = "#E0E0E0"
BLANC        = "#FFFFFF"
ORANGE       = "#E07B39"

CMAP_DIVERGE = mcolors.LinearSegmentedColormap.from_list(
    "cmap_diverge", [MAUVE_FONCE, VERT_PALE, VERT_FONCE]
)

CMAP_MISSING = mcolors.ListedColormap([VERT_PALE, MAUVE_FONCE])

plt.rcParams.update({
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.color":         GRIS_CLAIR,
    "grid.linewidth":     0.5,
    "figure.facecolor":   BLANC,
    "axes.facecolor":     BLANC,
})

# ─────────────────────────────────────────
# DONNÉES
# ─────────────────────────────────────────

df_imputation, df_model, df_famd_model = prepare_data()
y_var = "Overshoot_Day_DOY"

x_cols = [c for c in df_imputation.columns if c != y_var]

def profil_manque(row):
    n_x_missing = row[x_cols].isna().sum()
    y_missing   = pd.isna(row[y_var])
    if n_x_missing >= 10:
        return "bloc_systematique"
    elif n_x_missing > 0 and y_missing:
        return "na_x_et_y"
    elif n_x_missing > 0:
        return "na_x_partiel"
    elif y_missing:
        return "na_y"
    else:
        return "complet"

df_imputation["profil"] = df_imputation.apply(profil_manque, axis=1)

# ─────────────────────────────────────────
# CARTE
# ─────────────────────────────────────────

url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("/tmp/ne_110m")

world = gpd.read_file("/tmp/ne_110m/ne_110m_admin_0_countries.shp")

df_profil = df_imputation[["profil"]].reset_index()

def normalize_country(s):
    return (
        s.str.lower()
         .str.strip()
         .str.replace(r"[''',.]", "", regex=True)
         .str.replace(r"\s+", " ", regex=True)
         .str.replace(r"[^a-z ]", "", regex=True)
         .str.strip()
    )
world["name_clean"] = normalize_country(world["NAME"])
df_profil["name_clean"] = normalize_country(df_profil["Country"])

mapping_true_ne = {
    # Amériques
    "antigua and barbuda":                  "antigua and barb",
    "dominican republic":                   "dominican rep",
    "saint lucia":                          "st lucia",
    "venezuela bolivarian republic of":     "venezuela",
    "french guiana":                        "france",       # absent, rattaché à la France dans 110m

    # Europe
    "bosnia and herzegovina":               "bosnia and herz",
    "czech republic":                       "czechia",
    "republic of north macedonia":          "north macedonia",
    "republic of moldova":                  "moldova",
    "russian federation":                   "russia",

    # Afrique
    "cabo verde":                           "cape verde",
    "central african republic":             "central african rep",
    "congo democratic republic of": "dem rep congo",
    "congo rep":                            "congo",
    "equatorial guinea":                    "eq guinea",
    "libyan arab jamahiriya":               "libya",
    "sao tome and principe":                "sao tome and principe",
    "south sudan":                          "s sudan",
    "tanzania united republic of":          "tanzania",

    # Asie
    "brunei darussalam":                    "brunei",
    "iran islamic republic of":             "iran",
    "korea democratic peoples republic of": "north korea",
    "korea republic of":                    "south korea",
    "lao peoples democratic republic":      "laos",
    "syrian arab republic":                 "syria",
    "viet nam":                             "vietnam",

    # Pacifique / autres
    "solomon islands":                      "solomon is",
    "state of palestine":                   "palestine",
    
}

df_profil["name_clean"] = df_profil["name_clean"].replace(mapping_true_ne)

world = world.merge(
    df_profil[["name_clean", "profil"]],
    on="name_clean",
    how="left"
)

world["profil"] = world["profil"].fillna("non_disponible")

# ─────────────────────────────────────────
# PLOT CARTE
# ─────────────────────────────────────────

couleurs_carte = {
    "complet":           VERT_MOYEN,
    "na_y":              MAUVE_MOYEN,
    "na_x_partiel":      MAUVE_CLAIR,
    "na_x_et_y":         ORANGE,
    "bloc_systematique": MAUVE_FONCE,
    "non_disponible":    GRIS_CLAIR,
}

labels_carte = {
    "complet":           "X et Y complets: 106 pays",
    "na_y":              "Y manquant, X complet: 38 pays",
    "na_x_partiel":      "X partiellement manquant, Y observé: 10 pays",
    "na_x_et_y":         "X partiellement manquant, Y manquant: 8 pays",
    "bloc_systematique": "Bloc systématique manquant: 22 pays",
    "non_disponible":    "Non disponible",
}

world_robin = world.to_crs("+proj=robin")

fig, ax = plt.subplots(figsize=(16, 9))

for profil, color in couleurs_carte.items():
    world_robin[world_robin["profil"] == profil].plot(
        ax=ax, color=color, linewidth=0.3, edgecolor="white"
    )

patches = [
    mpatches.Patch(color=couleurs_carte[p], label=labels_carte[p])
    for p in couleurs_carte
]
ax.legend(
    handles=patches,
    loc="lower left",
    fontsize=8.5,
    framealpha=0.95,
    edgecolor=GRIS_CLAIR,
    title="Profil des données manquantes",
    title_fontsize=9,
)

ax.set_title(
    r"$\text{Répartition géographique des données manquantes par pays}$",
    fontsize=20,pad=14, color=VERT_FONCE, fontweight="bold"
)

ax.axis("off")
plt.tight_layout()
plt.savefig("fig1_carte_manque.png", dpi=150, bbox_inches="tight")
# ─────────────────────────────────────────
# HEATMAP NA (NON TRIÉE)
# ─────────────────────────────────────────

cols_plot = [c for c in df_imputation.columns
             if c not in [y_var, "Income_Group_Code", "Quality_Score_Code", "profil"]]

df_plot = df_imputation[cols_plot].copy()

df_plot = df_plot.rename(columns=NOMS_VARIABLES)

df_plot.columns = [
    r"$\mathrm{" + c.replace("_", r"\_").replace(" ", r"\ ") + "}$"
    for c in df_plot.columns
]

na_df = df_plot.isna().astype(int).T

fig, ax = plt.subplots(figsize=(14, 6))

sns.heatmap(
    na_df,
    cmap=["#DDEADF", MAUVE_FONCE],
    cbar=False,
    xticklabels=False,
    yticklabels=True,
    linewidths=0.15,
    linecolor="white",
    ax=ax
)

ax.set_title(r"$\text{Structure des données manquantes}$", fontsize=20,color=VERT_FONCE, fontweight="bold")
ax.set_xlabel(r"$\text{Pays}$", fontweight="bold")
ax.xaxis.set_label_position('top')
ax.set_ylabel(r"$\text{Variables}$", fontweight="bold")

plt.tight_layout()
plt.savefig("fig2_heatmap_na.png", dpi=150)


# ─────────────────────────────────────────
# CORRÉLATION NA (FIX)
# ─────────────────────────────────────────
ax.grid(False)
fig, ax = plt.subplots(figsize=(14, 11))
ax.grid(False)

df_plot_latex = df_plot.copy()
msno.heatmap(
    df_plot_latex, ax=ax,
    fontsize=8, cmap=CMAP_DIVERGE
)
ax.set_title(
    r"$\text{Corrélation des valeurs manquantes entre variables}$",
    fontsize=20,pad=10, color=VERT_FONCE, fontweight="bold"
)
plt.tight_layout(pad=3, w_pad=2, h_pad=2)
plt.savefig("fig3_correlation_na.png", dpi=150)


## ─────────────────────────────────────────
## IV. Construction de l'échantillon d'apprentissage
## ─────────────────────────────────────────

from split import split

coords = pd.read_csv("famd_model_coords.csv").set_index("Country")

train_idx, test_idx = split(df_model, coords, p=0.7, k=10, seed=123)

common = df_model.index.intersection(coords.index)
df_model_plot = df_model.loc[common].copy()
coords_plot = coords.loc[common].copy()

missing_y_countries = df_model[df_model[y_var].isna()].index

# ── Figure 4 — Biplot FAMD : Y observé vs Y manquant ───────────────────────

coords_plot["statut_y"] = np.where(
    df_model_plot[y_var].isna(),
    "Y manquant",
    "Y observé"
)

couleurs_y = {
    "Y observé":  VERT_MOYEN,
    "Y manquant": MAUVE_FONCE,
}

fig, ax = plt.subplots(figsize=(10, 7))
ax.grid(False)

for statut, group in coords_plot.groupby("statut_y"):
    ax.scatter(
        group["Dim.1"], group["Dim.2"],
        c=couleurs_y[statut],
        s=45, alpha=0.8, edgecolors="none",
        label=statut
    )

ax.axvline(0, color=GRIS_CLAIR, linewidth=0.8, linestyle="--")
ax.axhline(0, color=GRIS_CLAIR, linewidth=0.8, linestyle="--")
ax.set_xlabel(r"$\text{Dimension 1}$", fontsize=10, color=GRIS_FONCE, fontweight="bold")
ax.set_ylabel(r"$\text{Dimension 2}$", fontsize=10, color=GRIS_FONCE, fontweight="bold")
ax.set_title(
    r"$\text{Espace FAMD: disponibilité de } Y \text{ par pays}$",
    fontsize=16, pad=12, color=VERT_FONCE, fontweight="bold"
)
ax.legend(fontsize=9, framealpha=0.95, edgecolor=GRIS_CLAIR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig4_famd_y.png", dpi=150, bbox_inches="tight")

# ── Figure 5 — Split train/test sur le plan FAMD ───────────────────────────

def get_role(country):
    if country in train_idx:
        return "Apprentissage"
    elif country in test_idx:
        return "Test"
    elif country in missing_y_countries:
        return "Y manquant"
    else:
        return "Autre"

coords_plot["role"] = [get_role(c) for c in coords_plot.index]

couleurs_role = {
    "Apprentissage": VERT_MOYEN,
    "Test":          ORANGE,
    "Y manquant":    MAUVE_FONCE,
    "Autre":         GRIS_CLAIR,
}

tailles_role = {
    "Apprentissage": 45,
    "Test":          45,
    "Y manquant":    60,
    "Autre":         20,
}

fig, ax = plt.subplots(figsize=(10, 7))
ax.grid(False)

for role, group in coords_plot.groupby("role"):
    ax.scatter(
        group["Dim.1"], group["Dim.2"],
        c=couleurs_role[role],
        s=tailles_role[role],
        alpha=0.8, edgecolors="none",
        label=role
    )

ax.axvline(0, color=GRIS_CLAIR, linewidth=0.8, linestyle="--")
ax.axhline(0, color=GRIS_CLAIR, linewidth=0.8, linestyle="--")
ax.set_xlabel(r"$\text{Dimension 1}$", fontsize=10, color=GRIS_FONCE, fontweight="bold")
ax.set_ylabel(r"$\text{Dimension 2}$", fontsize=10, color=GRIS_FONCE, fontweight="bold")
ax.set_title(
    r"$\text{Espace FAMD: répartition apprentissage / test / } Y \text{ manquant}$",
    fontsize=20, pad=12, color=VERT_FONCE, fontweight="bold"
)
ax.legend(fontsize=9, framealpha=0.95, edgecolor=GRIS_CLAIR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig5_famd_split.png", dpi=150, bbox_inches="tight")

# counts pour vérification
print(coords_plot["role"].value_counts())
dim1 = coords["Dim.1"]
train_left  = [c for c in train_idx if dim1[c] < 0]
train_right = [c for c in train_idx if dim1[c] >= 0]
test_left   = [c for c in test_idx  if dim1[c] < 0]
test_right  = [c for c in test_idx  if dim1[c] >= 0]
print(f"Apprentissage — gauche : {len(train_left)}, droite : {len(train_right)}")
print(f"Test          — gauche : {len(test_left)},  droite : {len(test_right)}")


## ── Figure 6 — Distribution des poids KNN (pays observés à gauche de l'axe 1) ──

from sklearn.metrics.pairwise import euclidean_distances

dim_cols = [c for c in coords.columns if c.startswith("Dim.")]
X_coords = coords[dim_cols].values

dist = pd.DataFrame(
    euclidean_distances(X_coords),
    index=coords.index,
    columns=coords.index
)

bleu_gauche = coords.index[
    (coords["Dim.1"] < 0) &
    (df_model.loc[coords.index, y_var].notna())
]

missing_y = df_model[y_var].isna()

k = 10
poids = {}
for c in bleu_gauche:
    voisins = dist.loc[c].sort_values().index[1:k+1]
    n_manquant = missing_y.loc[voisins].sum()
    poids[c] = n_manquant / k

poids_series = pd.Series(poids).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
ax.grid(False)

couleurs_poids = [MAUVE_FONCE if p >= 0.5 else VERT_MOYEN for p in poids_series.values]

ax.bar(
    range(len(poids_series)),
    poids_series.values,
    color=couleurs_poids,
    edgecolor="none",
    width=0.8
)

ax.axhline(0.5, color=ORANGE, linewidth=1, linestyle="--", label="Seuil 0.5")

ax.set_xticks([])
ax.set_xlabel(
    r"$\text{Pays observés à gauche de l'axe 1 (triés par poids décroissant)}$",
    fontsize=9, color=GRIS_FONCE, fontweight="bold"
)
ax.set_ylabel(
    r"$\text{Proportion de voisins avec } Y \text{ manquant}$",
    fontsize=9, color=GRIS_FONCE, fontweight="bold"
)
ax.set_title(
    r"$\text{Poids de sélection en apprentissage: pays observés à gauche de l'axe FAMD}$",
    fontsize=20, pad=12, color=VERT_FONCE, fontweight="bold"
)
ax.set_ylim(0, 1.05)

patches = [
    mpatches.Patch(color=MAUVE_FONCE, label="Poids $\geq$ 0.5 (profil proche des pays manquants)"),
    mpatches.Patch(color=VERT_MOYEN,  label="Poids $<$ 0.5"),
    plt.Line2D([0], [0], color=ORANGE, linestyle="--", label="Seuil 0.5"),
]
ax.legend(handles=patches, fontsize=8.5, framealpha=0.95, edgecolor=GRIS_CLAIR)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("fig6_poids_knn.png", dpi=150, bbox_inches="tight")

print(f"Pays à gauche avec Y observé : {len(bleu_gauche)}")
print(f"Poids min : {poids_series.min():.2f}, max : {poids_series.max():.2f}, médiane : {poids_series.median():.2f}")


## ─────────────────────────────────────────
## I. Scénario 1
## ─────────────────────────────────────────

import joblib
import sys

from split import split

coords = pd.read_csv("famd_model_coords.csv").set_index("Country")
train_idx, test_idx = split(df_model, coords, p=0.7, k=10, seed=123)

df_obs   = df_model[df_model[y_var].notna()].copy()
df_train = df_obs.loc[train_idx]
df_test  = df_obs.loc[test_idx]
X_train  = df_train.drop(columns=[y_var])
y_train  = df_train[y_var]
X_test   = df_test.drop(columns=[y_var])
y_test   = df_test[y_var]

rf            = joblib.load("rf_model.pkl")
selected_vars = joblib.load("selected_vars.pkl")

y_pred_test  = rf.predict(X_test[selected_vars])
y_pred_train = rf.predict(X_train[selected_vars])

# ── Figure 7 — Scatter prédit vs réel ──────────────────────────────────────

min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())

fig, ax = plt.subplots(figsize=(7, 7))
ax.grid(False)

ax.scatter(
    y_test, y_pred_test,
    color=VERT_MOYEN, alpha=0.8, s=50, edgecolors="none"
)
ax.plot(
    [min_val, max_val], [min_val, max_val],
    color=MAUVE_FONCE, linewidth=1.2, linestyle="--", label="$y = x$"
)

ax.set_xlabel(r"$\text{Valeur réelle (jours)}$", fontsize=10, color=GRIS_FONCE, fontweight="bold")
ax.set_ylabel(r"$\text{Valeur prédite (jours)}$", fontsize=10, color=GRIS_FONCE, fontweight="bold")
ax.set_title(
    r"$\text{Prédit vs réel: forêt aléatoire}$",
    fontsize=16, pad=12, color=VERT_FONCE, fontweight="bold"
)
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig7_predit_vs_reel.png", dpi=150, bbox_inches="tight")
plt.show()

# ── Figure 8 — Importance des variables ────────────────────────────────────

importances = pd.Series(
    rf.feature_importances_,
    index=selected_vars
).sort_values(ascending=True).tail(20)

importances.index = [NOMS_VARIABLES.get(v, v) for v in importances.index]

fig, ax = plt.subplots(figsize=(9, 7))
ax.grid(False)

bars = ax.barh(
    importances.index,
    importances.values,
    color=VERT_MOYEN,
    edgecolor="none"
)

# top 5 en mauve
for i, (val, bar) in enumerate(zip(importances.values, bars)):
    if i >= len(importances) - 5:
        bar.set_color(MAUVE_FONCE)

ax.set_xlabel(
    r"$\text{Importance (impureté de Gini moyenne)}$",
    fontsize=9, color=GRIS_FONCE, fontweight="bold"
)
ax.set_title(
    r"$\text{Importance des variables: forêt aléatoire (top 20)}$",
    fontsize=20, pad=12, color=VERT_FONCE, fontweight="bold"
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig("fig8_importance_vars.png", dpi=150, bbox_inches="tight")
plt.show()

## ── Figure 9 — Pays triés par jour de dépassement prédit ──────────────────

rf_full_model = joblib.load("rf_model.pkl")
selected_vars_full = joblib.load("selected_vars.pkl")

df_all = df_model.copy()
common = df_all.index.intersection(coords.index)
df_all = df_all.loc[common]

df_all["y_pred"] = rf_full_model.predict(df_all[selected_vars_full])

df_all["statut"] = np.where(
    df_all[y_var].isna(),
    "Y manquant",
    "Y observé"
)

df_all["y_plot"] = np.where(
    df_all[y_var].isna(),
    df_all["y_pred"],
    df_all[y_var]
)

df_sorted = df_all.sort_values("y_plot").reset_index()
mediane_obs = df_all.loc[df_all["statut"] == "Y observé", y_var].median()

couleurs_statut = {
    "Y observé":  VERT_MOYEN,
    "Y manquant": MAUVE_MOYEN,
}

fig, ax = plt.subplots(figsize=(20, 7))
ax.grid(False)

for _, row in df_sorted.iterrows():
    ax.scatter(
        row["Country"], row["y_plot"],
        color=couleurs_statut[row["statut"]],
        s=30, alpha=0.85, edgecolors="none", zorder=3
    )

ax.axhline(
    mediane_obs,
    color=ORANGE, linewidth=1, linestyle="--",
    label=f"Médiane observés : {int(mediane_obs)} jours"
)

ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(
    [NOMS_PAYS.get(c, c) for c in df_sorted["Country"]],
    rotation=90, fontsize=5.5, color=GRIS_FONCE
)

ax.set_ylabel(
    r"$\text{Jour de dépassement (DOY)}$",
    fontsize=10, color=GRIS_FONCE, fontweight="bold"
)
ax.set_xlabel(r"$\text{Pays}$", fontsize=10, color=GRIS_FONCE, fontweight="bold")
ax.set_title(
    r"$\text{Jour de dépassement prédit}$",
    fontsize=20, pad=12, color=VERT_FONCE, fontweight="bold"
)

patches = [
    mpatches.Patch(color=VERT_MOYEN,  label="$Y$ observé"),
    mpatches.Patch(color=MAUVE_MOYEN, label="$Y$ manquant (prédit)"),
    plt.Line2D([0], [0], color=ORANGE, linestyle="--",
            label=f"Médiane observés : {int(mediane_obs)} jours"),
]
ax.legend(handles=patches, fontsize=9, framealpha=0.95, edgecolor=GRIS_CLAIR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("fig9_pays_tries.png", dpi=150, bbox_inches="tight")
plt.show()




## ─────────────────────────────────────────
## II. Scénario 2
## ─────────────────────────────────────────

from sklearn.metrics.pairwise import euclidean_distances

df_all_s2 = df_model.loc[common].copy()
df_all_s2["y_pred"] = rf.predict(df_all_s2[selected_vars])

dim_cols = [c for c in coords.columns if c.startswith("Dim.")]
dist_s2 = pd.DataFrame(
    euclidean_distances(coords[dim_cols].values),
    index=coords.index,
    columns=coords.index
)

k = 5
knn_df = []

for country in df_all_s2.index:
    neighbors = dist_s2.loc[country].sort_values().index[1:k+1]
    neigh_obs = [
        df_all_s2.loc[n, y_var]
        for n in neighbors
        if not pd.isna(df_all_s2.loc[n, y_var])
    ]
    if len(neigh_obs) == 0:
        continue
    knn_df.append({
        "Country":   country,
        "y_pred":    df_all_s2.loc[country, "y_pred"],
        "knn_mean":  np.mean(neigh_obs),
        "statut":    "Y manquant" if pd.isna(df_all_s2.loc[country, y_var]) else "Y observé"
    })

knn_df = pd.DataFrame(knn_df)

# ── Figure 10 — Scatter cohérence KNN ──────────────────────────────────────

min_val = min(knn_df["knn_mean"].min(), knn_df["y_pred"].min())
max_val = max(knn_df["knn_mean"].max(), knn_df["y_pred"].max())

couleurs_knn = {
    "Y observé":  VERT_MOYEN,
    "Y manquant": MAUVE_MOYEN,
}

fig, ax = plt.subplots(figsize=(8, 8))
ax.grid(False)

for statut, group in knn_df.groupby("statut"):
    ax.scatter(
        group["knn_mean"], group["y_pred"],
        color=couleurs_knn[statut],
        s=40, alpha=0.8, edgecolors="none",
        label=statut
    )

ax.plot(
    [min_val, max_val], [min_val, max_val],
    color=MAUVE_FONCE, linewidth=1.2, linestyle="--", label="$y = x$"
)

ax.set_xlabel(
    r"$\text{Moyenne des } k \text{ plus proches voisins observés (jours)}$",
    fontsize=9, color=GRIS_FONCE, fontweight="bold"
)
ax.set_ylabel(
    r"$\text{Jour de dépassement prédit (jours)}$",
    fontsize=9, color=GRIS_FONCE, fontweight="bold"
)
ax.set_title(
    r"$\text{Cohérence locale des prédictions: prédit vs moyenne KNN}$",
    fontsize=16, pad=12, color=VERT_FONCE, fontweight="bold"
)
ax.legend(fontsize=9, framealpha=0.95, edgecolor=GRIS_CLAIR)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig10_knn_coherence.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Corrélation prédictions / moyenne KNN : {knn_df['y_pred'].corr(knn_df['knn_mean']):.3f}")

outlier = knn_df[
    (knn_df["statut"] == "Y manquant") &
    (knn_df["y_pred"] > 340) &
    (knn_df["knn_mean"] < 200)
]["Country"]
print(outlier)


## ─────────────────────────────────────────
## III. Scénario 3
## ─────────────────────────────────────────

import missingno as msno

df_avant = pd.read_csv("df_avant_amputation.csv", index_col=0)
df_apres = pd.read_csv("data/df_amputed.csv", index_col=0)

cols_num = [c for c in df_avant.columns
            if pd.api.types.is_numeric_dtype(df_avant[c])
            and c not in ["Overshoot_Day_DOY", "Income_Group_Code", "Quality_Score_Code"]]

# masque : cellules artificiellement amputées
masque = (df_apres[cols_num].isna() & df_avant[cols_num].notna())

# ── Figure 11 — Heatmap corrélation NA post-amputation ─────────────────────

df_apres_plot = df_apres[cols_num].copy()
df_apres_plot = df_apres_plot.rename(columns=NOMS_VARIABLES)
df_apres_plot.columns = [
    r"$\mathrm{" + c.replace("_", r"\_").replace(" ", r"\ ") + "}$"
    for c in df_apres_plot.columns
]

fig11, ax11 = plt.subplots(figsize=(14, 11))
ax11.grid(False)

msno.heatmap(
    df_apres_plot, ax=ax11,
    fontsize=8, cmap=CMAP_DIVERGE
)
ax11.set_title(
    r"$\text{Corrélation des valeurs manquantes après amputation}$",
    fontsize=20, pad=10, color=VERT_FONCE, fontweight="bold"
)
plt.tight_layout(pad=3, w_pad=2, h_pad=2)
plt.savefig("fig11_correlation_na_post_amputation.png", dpi=150)
plt.show()

# ── Figure 12 — Dendrogramme ───────────────────────────────────────────────

fig12, ax12 = plt.subplots(figsize=(14, 8))
ax12.grid(False)

msno.dendrogram(
    df_apres_plot, ax=ax12,
    fontsize=8
)
ax12.set_title(
    r"$\text{Dendrogramme des valeurs manquantes: jeu de données amputé}$",
    fontsize=20, pad=10, color=VERT_FONCE, fontweight="bold"
)
plt.tight_layout(pad=3)
plt.savefig("fig12_dendrogramme.png", dpi=150)
plt.show()