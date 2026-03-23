import pandas as pd
import numpy as np
import math
import scipy.stats
from sklearn.metrics.pairwise import euclidean_distances
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# En tout il y a 144 pays
# 38 pays ont Y manquants
# Dans notre train et notre test du S1 =, nous ne les voulons pas donc n = 106
# On prend 70% pour le train donc : 
# n_train = 75 et n_test = 31

# =========================================================
# HYPERPARAMETRES
# =========================================================
p    = 0.7   # proportion train
k    = 10    # nombre de voisins pour le calcul des poids
SEED = 123   # graine pour la reproductibilité et train fixe
np.random.seed(SEED)

# =========================================================
# 1. CHARGEMENT
# =========================================================

df_model = pd.read_csv("df_model.csv", index_col=0)
coords = pd.read_csv("famd_model_coords.csv")
coords = coords.set_index("Country")

common = df_model.index.intersection(coords.index)
df_model = df_model.loc[common].copy()
coords = coords.loc[common].copy()

y_var = "Overshoot_Day_DOY"

# =========================================================
# 2. FLAGS Y OBSERVE / MANQUANT
# =========================================================

coords["Y_observe"] = df_model[y_var].notna()
coords["Y_manquant"] = df_model[y_var].isna()
coords["cote"] = np.where(coords["Dim.1"] < 0, "gauche", "droite")

# =========================================================
# 3. MATRICE DE DISTANCES DANS L'ESPACE FAMD
# =========================================================

dim_cols = [c for c in coords.columns if c.startswith("Dim.")]
X = coords[dim_cols].values

dist = pd.DataFrame(
    euclidean_distances(X),
    index=coords.index,
    columns=coords.index
)

# =========================================================
# 4. COMPLETS A GAUCHE ET A DROITE
# =========================================================

bleu_gauche = coords.index[
    (coords["cote"] == "gauche") &
    (coords["Y_observe"])
]

n_bleu_gauche = len(bleu_gauche) 

bleu_droite = coords.index[
    (coords["cote"] == "droite") &
    (coords["Y_observe"])
]

n_bleu_droite = len(bleu_droite) 

print(f"Complets gauche : {n_bleu_gauche} pays")
print(f"Complets droite : {n_bleu_droite} pays")
print(f"Manquants   : {coords['Y_manquant'].sum()} pays")
print(f"Total       : {len(coords)} pays")


# ===========================================================
# 5. VISUALISATION AVEC plotly, POUR L INTERACIVITE
# ===========================================================

fig = make_subplots(rows=1, cols=3, subplot_titles=[
    "Espace FAMD - Base",
    "GAUCHE",
    "DROITE"
])

# Fonction utilitaire
def make_scatter(subset, color, name, symbol="circle", size=6):
    return go.Scatter(
        x=coords.loc[subset, "Dim.1"],
        y=coords.loc[subset, "Dim.2"],
        mode="markers",
        marker=dict(color=color, size=size, symbol=symbol),
        text=subset,
        hovertemplate="<b>%{text}</b><br>Dim.1: %{x:.2f}<br>Dim.2: %{y:.2f}<extra></extra>",
        name=name
    )

# Graphique 1 : Base
fig.add_trace(make_scatter(coords.index[coords["Y_observe"]], "steelblue", "Observés"), row=1, col=1)
fig.add_trace(make_scatter(coords.index[coords["Y_manquant"]], "tomato",    "Manquants"), row=1, col=1)

# Graphique 2 : Gauche
fig.add_trace(make_scatter(coords.index, "lightgrey", "Tous", size=5), row=1, col=2)
fig.add_trace(make_scatter(bleu_gauche,  "cornflowerblue", "Observés gauche"), row=1, col=2)
fig.add_trace(make_scatter(coords.index[coords["Y_manquant"]], "tomato", "Manquants"), row=1, col=2)

# Graphique 3 : Droite
fig.add_trace(make_scatter(coords.index, "lightgrey", "Tous", size=5), row=1, col=3)
fig.add_trace(make_scatter(bleu_droite,  "seagreen", "Observés droite"), row=1, col=3)
fig.add_trace(make_scatter(coords.index[coords["Y_manquant"]], "tomato", "Manquants"), row=1, col=3)

# Ligne verticale à 0 sur chaque graphique
for col in [1, 2, 3]:
    fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1, col=col, row=1)

fig.update_layout(
    title_text="Espace FAMD — Sélection des pays de référence",
    title_font_size=14,
    height=500,
    width=1200,
    showlegend=False
)

fig.show()

# =========================================================
# 6. VECTEURS DE POIDS/PROBA PAR PAYS DE GAUCHE
# =========================================================

p_dict_all = {}
for c in bleu_gauche:
    d = dist.loc[c].sort_values()
    voisins = d.index[1:k+1]
    n_manquant = coords.loc[voisins, "Y_manquant"].sum()
    p_dict_all[c] = n_manquant / k

poids_all = pd.Series(p_dict_all)
# print("\nVecteur de poids :")
# for i, (pays, val) in enumerate(poids_all.items(), start=1):
#    print(f"{i:>3}. {pays:<30} {val:.4f}")

proba_all = poids_all / poids_all.sum()
# print("\nVecteur de poids normalisé, proba :")
# for i, (pays, val) in enumerate(proba_all.items(), start=1):
#    print(f"{i:>3}. {pays:<30} {val:.4f}")

print(f"\nVérification somme des probas du vecteur de poids normalisé : {proba_all.sum():.6f}")

n_gauche_70 = math.ceil(n_bleu_gauche * p)

# ================================================================
# 7. SELECTION DE 70% A GAUCHE (multinomiale) 
# ================================================================

# on a choisis d'utiliser np.random.choice plutot que np.random.multinomial car avec elle il y a une remise et donc un risque de prendre 2 fois le même pays,
# c'est ce qui s'est passé avec le code suivant, plusieurs pays étaient pris plusieurs fois

# counts = scipy.stats.multinomial.rvs(n=n_gauche_70, p=proba_all.loc[bleu_gauche].values)
# bleu_gauche_70 = pd.Index(
#    np.repeat(bleu_gauche, counts)  # répète chaque pays selon son comptage
#)

bleu_gauche_70 = pd.Index(
    np.random.choice(bleu_gauche, size=n_gauche_70, replace=False, p=proba_all.loc[bleu_gauche].values)
)

print(f"Total bleu gauche : {n_bleu_gauche} pays")
print(f"70% sélectionnés gauche (multinomiale) : {len(bleu_gauche_70)} pays")
for i, pays in enumerate(bleu_gauche_70, start=1):
    print(f"{i:>3}. {pays}")

# ================================================================
# 8. SELECTION DE 70% DROITE (uniforme)
# ================================================================

# --- DROITE : tirage uniforme ---
n_droite_70 = math.ceil(n_bleu_droite * p)

bleu_droite_70 = pd.Index(
    np.random.choice(bleu_droite, size=n_droite_70, replace=False)  # uniforme par défaut
)

print(f"\nTotal bleu droite : {n_bleu_droite} pays")
print(f"70% sélectionnés droite (uniforme)     : {len(bleu_droite_70)} pays")
# for i, pays in enumerate(bleu_droite_70, start=1):
#    print(f"{i:>3}. {pays}")

# ================================================================
# 9. CREATION DES TRAIN / TEST
# ================================================================

train = bleu_gauche_70.union(bleu_droite_70)
test  = coords.index[coords["Y_observe"]].difference(train)

print(f"Train : {len(train)} pays {train}")
print(f"Test  : {len(test)} pays {test}")

# ================================================================
# 10. VISUALISATION DES TRAIN ET TEST
# ================================================================

fig2 = make_subplots(rows=1, cols=2, subplot_titles=[
    "70% sélectionnés (Gauche + Droite)",
    "Train / Test / Manquants"
])

# Graphique 1 : Gauche + Droite 70%
for trace in [
    make_scatter(coords.index,                        "lightgrey",      "Tous",      size=5),
    make_scatter(bleu_gauche_70,                      "cornflowerblue", "Gauche 70%"),
    make_scatter(bleu_droite_70,                      "seagreen",       "Droite 70%"),
    make_scatter(coords.index[coords["Y_manquant"]], "tomato",         "Manquants"),
]:
    fig2.add_trace(trace, row=1, col=1)

# Graphique 2 : Train / Test / Manquants
for trace in [
    make_scatter(coords.index,                        "lightgrey",  "Tous",       size=5),
    make_scatter(train,                               "mediumorchid","Train (70%)"),
    make_scatter(test,                                "gold",        "Test (30%)"),
    make_scatter(coords.index[coords["Y_manquant"]], "tomato",      "Manquants"),
]:
    fig2.add_trace(trace, row=1, col=2)

# Lignes verticales
for col in [1, 2]:
    fig2.add_vline(x=0, line_dash="dash", line_color="black", line_width=1, col=col, row=1)

fig2.update_layout(
    title_text="Espace FAMD — Sélection et Split",
    title_font_size=14,
    height=500,
    width=1200,
    showlegend=True,
    xaxis_title="Dim.1",  yaxis_title="Dim.2",
    xaxis2_title="Dim.1", yaxis2_title="Dim.2"
)

fig2.show()

# ================================================================
# 11. EXPORT TRAIN / TEST
# ================================================================

df_model.loc[train].to_csv("train.csv")
df_model.loc[test].to_csv("test.csv")

print(f"train.csv exporté : {len(train)} pays")
print(f"test.csv exporté  : {len(test)} pays")