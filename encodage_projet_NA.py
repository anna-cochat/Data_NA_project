import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

pays = pd.read_excel("C:/Users/celin/Projet tutoré/juste_mes_pays.xlsx", header=0)
pays["Per Capita GDP"] = pays["Per Capita GDP"].replace("-", np.nan)
pays = pays.rename(columns={
    "actual \nCountry Overshoot Day \n2018": "Overshoot day"
})
pays = pays.dropna()


pays["Per Capita GDP"] = (
    pays["Per Capita GDP"]
    .astype(str)
    .str.replace(r"[^\d]", "", regex=True)    # garde seulement les chiffres
    .astype(int)
)
print(pays["Per Capita GDP"])
donnees_quali=pays[["Quality Score",  "Region", "Income Group"]]
colonne_retir= ["Country", "Income Group", "Region", "Quality Score", "Overshoot day"]
pays_quanti= pays.drop(columns=colonne_retir)


def encodage(donneesquali):
    mat=pd.DataFrame()
    for e in donneesquali : #on fait l'encodage pour chaques questions
        liste=donneesquali[e].unique() #je regarde pour chaques questions le nombre de réponses possible
        for val in liste:
            col_enc=np.zeros(len(donneesquali)) #je crée au fur et a mesure les colonnes selon les réponses possible
            col_name= f"{e}_{val}" #c'est uniquement pour que mat_enc soit un peu plus lisible + pour ensuite vérifier que mon encodage marche juste en lisant ma matrice
            for k in range(len(donneesquali)):
                if donneesquali[e].iloc[k]==val: #je met un 1 dans la colonne possibilité qui correspond a la réponse, la ligne garde des 0 pour les autres colonnes
                    col_enc[k]=1
            mat[col_name]= col_enc #je raccroche colonne par colonne mon encodage a ma dataframe initialement vide en lui mettant aussi le nom adapté
    return(mat) #je retourne mon datafram contenant mes données qualitatives maintenant encodées

encod_pays=encodage(donnees_quali)
pays_fin= pd.concat([pays_quanti, encod_pays], axis=1)

print(pays_fin.head())