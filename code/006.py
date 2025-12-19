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

