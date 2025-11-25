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