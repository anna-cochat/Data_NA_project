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
