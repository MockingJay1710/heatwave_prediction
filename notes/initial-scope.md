L’objectif final du projet est de construire un modèle capable d’anticiper des épisodes de chaleur importante, puis progressivement d’aller vers une véritable prédiction de vague de chaleur.

# 1. Première question : que voulons-nous prédire ?

Le terme _prédiction de vague de chaleur_ est encore trop vague pour être directement utilisé comme cible d’un modèle.

Une vague de chaleur dépend notamment :
- de la température atteinte ;
- de la durée de l’épisode ;
- des températures minimales nocturnes ;
- de seuils qui peuvent dépendre de la région considérée.

Pour commencer, nous choisissons donc un objectif plus simple :
> **Prédire la température maximale du jour suivant pour une localisation donnée.**

On cherchera donc à estimer : $T_{\max}(t+1)$ à partir des informations disponibles jusqu’au jour _t_.

## Pourquoi commencer par la température maximale du lendemain ?

- la temperature maximale est une variable continue, les modeles de regression lineaire sont sont naturellement adaptés a ce type de probleme
- on evite pour le moment de transformer le sujet en un probleme de classification : `vague_de_chaleur = oui ou non`
- on obtient plus d'information a partir d'une valeur numerique qu'a travers une classe

# 2. Premier modèle : une régression linéaire

Nous avons choisi de commencer avec une **régression linéaire**.
Elle nous permettra notamment de nous poser des questions simples :
- quelles variables semblent apporter de l’information ?
- certaines variables sont-elles redondantes ?
- jusqu’où peut-on aller avec un modèle très simple ?

# 3. Avant le modèle : établir une baseline

Avant même la régression linéaire, nous voulons disposer d’une référence extrêmement simple.
La première baseline sera :  $\hat{T}_{\max}(t+1) = T_{\max}(t)$

Autrement dit :
> **la température maximale de demain sera égale à celle d’aujourd’hui.**

Cela peut sembler naïf, mais la température possède une forte continuité temporelle.

Cette baseline est essentielle : si notre modèle de Machine Learning ne fait pas mieux, cela signifie que la complexité ajoutée n’est pas encore utile.

# 4. Première simplification spatiale : une seule localisation

À long terme, la météo dépend évidemment de ce qui se passe autour de la ville étudiée.
Une masse d’air chaude ou froide peut arriver depuis une région voisine, le vent transporte les masses d’air, et les systèmes atmosphériques sont spatiaux.
Cependant, pour la première étape, nous choisissons volontairement :

> **une seule localisation géographique.**

Nous récupérons donc les données Open-Meteo correspondant aux coordonnées de la ville ou du point étudié, **Nantes** dans notre cas.

# 5. Source des données : Open-Meteo

Nous avons choisi d’utiliser **Open-Meteo** pour récupérer les données météorologiques historiques.

Il faut garder en tête que ces valeurs proviennent de données météorologiques maillées / de réanalyse : la coordonnée demandée correspond donc à une cellule de modèle météorologique appropriée, et non nécessairement à un thermomètre physique exactement placé au point demandé.

Pour notre première étude, cette précision est suffisante.

- **`temperature_2m`** : Température de l’air à 2 mètres du sol. C’est la variable centrale de notre problème

- **`relative_humidity_2m`** — Humidité relative de l’air à 2 mètres, en %. Elle indique à quel point l’air est proche de la saturation en vapeur d’eau. On la garde car l’humidité caractérise l’état de l’atmosphère et peut aider à distinguer différents régimes météorologiques associés à la chaleur.

- **`pressure_msl`** — Pression atmosphérique ramenée au niveau de la mer. Elle permet de caractériser les systèmes météorologiques et les changements de masse d’air.

- **`cloud_cover`** — Pourcentage du ciel couvert par des nuages. Les nuages modifient fortement la quantité de rayonnement solaire qui atteint la surface.

- **`shortwave_radiation`** — Rayonnement solaire à ondes courtes reçu à la surface, en `W/m²`. C’est une information directement liée à l’énergie solaire disponible pour chauffer le sol et l’air.

- **`wind_speed_10m`** — Vitesse du vent à 10 mètres du sol. Le vent transporte des masses d’air et peut donc provoquer des changements de température.

- **`wind_direction_10m`** — Direction d’où vient le vent, en degrés. Elle complète la vitesse : un vent venant de l’Atlantique et un vent venant de l’intérieur des terres n’apportent pas nécessairement le même type de masse d’air.

- **`precipitation`** — Quantité de précipitations. La pluie renseigne sur le régime météorologique récent et influence aussi l’humidité du sol et les échanges d’énergie avec l’atmosphère.

- **`soil_moisture_0_to_7cm`** — Humidité de la couche superficielle du sol, entre 0 et 7 cm. Un sol humide peut utiliser une partie de l’énergie reçue pour l’évaporation, alors qu’un sol très sec peut davantage favoriser le chauffage de la surface.

