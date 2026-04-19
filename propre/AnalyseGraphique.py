import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("/Users/admin/Documents/GitHub/Data_NA_project/propre/toutlespays.xlsx")

# STYLE
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


if __name__ == "__main__":

    # sélectionner colonnes numériques
    numeric_cols = df.select_dtypes(include='number').columns

    # histogrammes pour toutes les variables
    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True, bins=20, color="#1B5E20")
        plt.title(f"Distribution de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.show()