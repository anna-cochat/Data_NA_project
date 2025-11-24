# HDI vs Nombre de NA

**Ce qu’on voit :**

- **Trois pays explosent les compteurs de NA** : _Cambodia, Cabo Verde, Antigua & Barbuda_.
    
- Ils ont un HDI **moyen à faible**, mais ce n’est pas ça qui explique vraiment leurs NA.
    
- La majorité des pays, même avec HDI bas (ex. Afghanistan, Burundi), **n’ont presque aucun NA**.

-> Le HDI n’explique pas la missingness. Ce sont **des cas spécifiques**, pas une tendance globale.  On est sûrement face à **problèmes de reporting / données nationales** propres à ces pays, pas un mécanisme structurel lié au développement. A verifier sur le dataset complet.

---

# PIB par habitant vs Nombre de NA

**Ce qu’on voit :**

- Les trois mêmes outliers ressortent **indépendamment du PIB**.
    
- Des pays très pauvres (Burundi, Niger) → **0–1 NA**.
    
- Des pays plus riches → **0–1 NA** aussi.
    
- Rien ne suit une vraie tendance.
    

-> Le PIB non plus n’explique pas la missingness. Les NA ne suivent pas la richesse du pays : donc **pas un biais systémique économique**. Idem à vérifier sur le dataset complet

---

# Boxplot NA par groupe de revenu

**Ce qu’on voit :**

- Tous les groupes → **manque très faible**, quasiment 0 NA.
    
- Les seuls points extrêmes = encore les 3 mêmes pays.
    
- Pas de différence visible entre LI / LM / UM / HI.

-> La missingness n’est PAS expliquée par les catégories de revenu. Idem. 

---

# Matrice de corrélation

**Ce qu’on voit :**

- HDI, PIB, Life Expectancy → très corrélés entre eux (logique).
    
- **NA_count n’a aucune corrélation** avec ces variables.
    
    - HDI → -0.13
        
    - PIB → -0.14
        
    - Life Expectancy → 0.01
        

-> La qualité du développement humain n’impacte pas la présence de NA.

---

# Heatmaps — Footprint Production / Consumption / Biocapacity

**Ce qu’on voit :**

- Les heatmaps montrent que **les NA sont concentrés exactement dans les mêmes pays et sur les mêmes variables** :
    
    - Antigua & Barbuda
        
    - Cabo Verde
        
    - Cambodia
        
- Sur TOUTES les catégories : Production, Consumption, Biocapacity.

-> La missingness n’est pas variable-spécifique : elle est pays-spécifique._  
Ces pays **ne reportent pas** leurs données écologiques dans plusieurs registres → c’est **une faille institutionnelle de reporting**, pas un biais du dataset.

---

# en gros

- La missingness est **faible** dans presque tout le sample.
    
- Elle se concentre sur **3 pays** qui manquent _massivement_ de données.
    
- Aucune variable socio-éco (HDI, PIB, revenu) n’explique ces manques.
    
- Les NA touchent toutes les dimensions écologiques : production, consommation, biocapacité.
    
- **Conclusion : le problème vient du reporting national**, pas de conditions socio-économiques ni du modèle de développement.
    

causes potentielles :

- Faiblesse institutionnelle,
    
- Failles administratives,
    
- Données non déclarées,
    
- Ou incompatibilité dans les formats nationaux.
    

see Angle 5 Gouvernance
see Country Profile for more details