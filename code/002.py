from preprocessing import load_mydf 
import matplotlib.pyplot as plt
import os
import seaborn as sns

mydf = load_mydf()

FIG_DIR = "figures/002"

def savefig(name):

    path = os.path.join(FIG_DIR, f"{name}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved: {path}") # enregistre les graphiques automatiquement

# On garde seulement les lignes où Footprint conso ET Biocapacité sont présents
df_angle2 = mydf.dropna(subset=["Total_Footprint_Consumption", "Total_Biocapacity"]).copy()
# si un des deux manque, on ne peut pas comparer conso et biocapacité, donc on retire


# On calcule l'équilibre écologique : Biocapacité - Empreinte conso
df_angle2["Eco_Balance"] = df_angle2["Total_Biocapacity"] - df_angle2["Total_Footprint_Consumption"]
# positif = réserve écologique
# négatif = déficit écologique


# 3. On détecte les outliers selon la méthode de l’IQR
Q1 = df_angle2["Eco_Balance"].quantile(0.25)  # premier quartile
Q3 = df_angle2["Eco_Balance"].quantile(0.75)  # troisième quartile
IQR = Q3 - Q1                                 # écart interquartile
lower = Q1 - 1.5 * IQR                        # borne basse outliers
upper = Q3 + 1.5 * IQR                        # borne haute outliers

outliers = df_angle2[(df_angle2["Eco_Balance"] < lower) | (df_angle2["Eco_Balance"] > upper)]
# pays très en déficit ou très en excédent écologique

#  Scatterplot conso vs biocapacité (plus la ligne d’équilibre)
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=df_angle2,
    x="Total_Footprint_Consumption",          #  empreinte de consommation
    y="Total_Biocapacity",                    #  biocapacité totale
    s=df_angle2["Population (millions)"] * 5, # taille du point = population (donne contexte démographique)
    color="darkred",
    alpha=0.7
)

plt.xlabel("Empreinte écologique (consommation)")
plt.ylabel("Biocapacité totale")
plt.title("Relation entre Empreinte de consommation et Biocapacité")

# trace la droite d’équilibre (biocapacité = empreinte)
max_val = max(df_angle2["Total_Footprint_Consumption"].max(),
    df_angle2["Total_Biocapacity"].max())
plt.plot([0, max_val], [0, max_val], color="gray", linestyle="--", label="Équilibre")

# on annote seulement les outliers
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
savefig("002_CFPvsBIOCP")  # enregistre automatiquement
plt.show()


# Barplot de la différence (Eco_Balance)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_angle2.sort_values("Eco_Balance"), # classement du plus grand déficit au plus grand surplus
    x="Country",
    y="Eco_Balance",
    palette=["darkred" if x < 0 else "forestgreen" 
    for x in df_angle2.sort_values("Eco_Balance")["Eco_Balance"]]
    # rouge = déficit écologique, vert = réserve
)

plt.xticks(rotation=90, fontsize=8)
plt.ylabel("Biocapacité – Empreinte (équilibre positif)")
plt.title("Différence entre Biocapacité et Empreinte écologique")
plt.tight_layout()
savefig("002_BarplotBalance")
plt.show()

#Histogramme distribution des réserves/déficits

plt.figure(figsize=(10, 6))
sns.histplot(df_angle2["Eco_Balance"], bins=10, kde=True, color="steelblue")
plt.axvline(0, color="black", linestyle="--")  # ligne = seuil déficit / réserve
plt.title("Distribution des déficits / réserves écologiques")
plt.xlabel("Biocapacité – Empreinte")
plt.ylabel("Nombre de pays")
plt.tight_layout()
savefig("002_HistBalance")
plt.show()

# Impression des outliers
print("\n Outliers")
print(outliers[["Country", "Total_Footprint_Consumption", "Total_Biocapacity", "Eco_Balance"]])
print("\nSeuils utilisés (IQR):")
print(f"Borne basse = {lower:.2f}")
print(f"Borne haute = {upper:.2f}")
