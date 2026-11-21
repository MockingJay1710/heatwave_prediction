



## 1. Objectif général du projet

L’objectif final du projet est de construire un modèle capable d’anticiper des épisodes de chaleur importante, puis progressivement d’aller vers une véritable prédiction de vague de chaleur.

Cependant, nous ne voulons pas commencer directement avec un modèle complexe ni avec un grand nombre de variables et de zones géographiques.  
Notre stratégie est volontairement progressive :

1. commencer avec un problème simple et clairement défini ;
2. établir un premier modèle de référence ;
3. vérifier ce que nous arrivons réellement à prédire ;
4. ajouter de nouvelles informations uniquement lorsqu’elles sont justifiées ;
5. mesurer à chaque étape si elles améliorent réellement les performances.

L’idée principale est donc de **comprendre progressivement le problème avant d’augmenter la complexité du modèle**.

---

# 2. Première question : que voulons-nous prédire ?

Le terme _prédiction de vague de chaleur_ est encore trop vague pour être directement utilisé comme cible d’un modèle.

Une vague de chaleur dépend notamment :

- de la température atteinte ;
- de la durée de l’épisode ;
- des températures minimales nocturnes ;
- de seuils qui peuvent dépendre de la région considérée.

Pour commencer, nous choisissons donc un objectif plus simple :

> **Prédire la température maximale du jour suivant pour une localisation donnée.**

On cherchera donc à estimer :

\[ T_{\max}(t+1) \]

à partir des informations disponibles jusqu’au jour \(t\).

## Pourquoi commencer par la température maximale du lendemain ?

Ce choix présente plusieurs avantages.

### C’est une variable continue

Une régression linéaire est naturellement adaptée à la prédiction d’une valeur numérique comme une température.

Nous évitons donc, pour l’instant, de transformer le problème en classification du type :

```
vague de chaleur = oui / non
```

qui demanderait immédiatement de choisir des seuils et une définition précise de l’événement.

### La température maximale est directement liée au problème final

Même si notre objectif final concerne les vagues de chaleur, celles-ci sont fortement liées à la persistance de températures maximales élevées.

Être capable de prédire correctement les températures maximales est donc une première brique utile.

### Cela donne plus d’information qu’une simple classe

Une prédiction comme :

```
37.4 °C
```

contient plus d’information que :

```
vague de chaleur : oui
```

Par la suite, les températures prédites pourront être utilisées pour détecter des dépassements de seuils ou des épisodes de chaleur persistante.

---

# 3. Premier modèle : une régression linéaire

Nous avons choisi de commencer avec une **régression linéaire**.

L’objectif n’est pas de construire immédiatement le meilleur modèle possible.  
Le but est d’avoir un modèle :

- simple ;
- interprétable ;
- rapide à entraîner ;
- facile à comparer à des baselines ;
- utile pour comprendre les relations entre les variables.

La forme générale sera :

\[ \hat{T}_{\max}(t+1) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n \]

où les \(x_i\) représentent les informations météorologiques disponibles avant la prédiction.

La régression linéaire nous permettra notamment de nous poser des questions simples :

- quelles variables semblent apporter de l’information ?
- certaines variables sont-elles redondantes ?
- le modèle généralise-t-il sur des années qu’il n’a jamais vues ?
- jusqu’où peut-on aller avec un modèle très simple ?

---

# 4. Avant le modèle : établir une baseline

Avant même la régression linéaire, nous voulons disposer d’une référence extrêmement simple.

La première baseline sera :

\[ \hat{T}_{\max}(t+1) = T_{\max}(t) \]

Autrement dit :

> **la température maximale de demain sera égale à celle d’aujourd’hui.**

Cela peut sembler naïf, mais la température possède une forte continuité temporelle.

Cette baseline est essentielle : si notre modèle de Machine Learning ne fait pas mieux, cela signifie que la complexité ajoutée n’est pas encore utile.

Notre première question expérimentale sera donc :

> **Une régression linéaire utilisant plusieurs informations météorologiques fait-elle mieux que simplement utiliser la température de la veille ?**

---

# 5. Première simplification spatiale : une seule localisation

À long terme, la météo dépend évidemment de ce qui se passe autour de la ville étudiée.

Une masse d’air chaude ou froide peut arriver depuis une région voisine, le vent transporte les masses d’air, et les systèmes atmosphériques sont spatiaux.

Cependant, pour la première étape, nous choisissons volontairement :

> **une seule localisation géographique.**

Nous récupérons donc les données Open-Meteo correspondant aux coordonnées de la ville ou du point étudié.

## Pourquoi ne pas intégrer immédiatement les villes environnantes ?

Parce que nous voulons d’abord répondre à une question plus simple :

> **Quelle quantité d’information peut-on déjà obtenir à partir de l’historique météorologique local ?**

Si le modèle local fonctionne raisonnablement bien, nous pourrons ensuite mesurer précisément le gain apporté par l’ajout d’informations spatiales.

Cela permettra une progression logique :

```
1 localisation
      ↓
zone environnante
      ↓
région plus large
      ↓
modèle spatio-temporel
```

Nous évitons ainsi de mélanger dès le départ deux difficultés :

- la dynamique temporelle ;
- la dynamique spatiale.

---

# 6. Source des données : Open-Meteo

Nous avons choisi d’utiliser **Open-Meteo** pour récupérer les données météorologiques historiques.

Pour cette première phase, nous utiliserons les données correspondant à une seule latitude/longitude.

Il faut garder en tête que ces valeurs proviennent de données météorologiques maillées / de réanalyse : la coordonnée demandée correspond donc à une cellule de modèle météorologique appropriée, et non nécessairement à un thermomètre physique exactement placé au point demandé.

Pour notre première étude, cette précision est suffisante.

---

# 7. Choix des variables à collecter

Nous ne voulons pas récupérer uniquement la température.

La température future dépend d’autres caractéristiques de l’atmosphère.  
Cependant, nous ne voulons pas non plus collecter toutes les variables disponibles sans raison.

Nous retenons donc un premier ensemble relativement compact.

## Variables principales

### `temperature_2m`

C’est la variable centrale du projet.

Elle sera utilisée :

- pour reconstruire les températures maximales journalières ;
- comme historique de température ;
- pour créer des variables retardées (_lags_).

Par exemple :

```
Tmax(t)
Tmax(t-1)
Tmax(t-2)
```

---

### `relative_humidity_2m`

L’humidité relative apporte une information sur l’état de l’air.

Elle peut notamment différencier plusieurs situations météorologiques associées à des températures similaires.

Elle est aussi particulièrement pertinente dans le contexte des fortes chaleurs.

---

### `dew_point_2m`

Le point de rosée est une autre manière de représenter la quantité d’humidité dans l’air.

Nous choisissons de le collecter même s’il est lié à la température et à l’humidité relative.

La question de sa redondance sera étudiée plus tard.

C’est une idée importante :

> Nous pouvons collecter une variable avant de savoir si elle sera finalement conservée dans le modèle.

---

### `pressure_msl`

La pression atmosphérique peut donner une information sur les systèmes météorologiques présents.

Certaines périodes de chaleur persistante sont associées à des situations atmosphériques stables.

Pour notre premier modèle, nous utilisons la pression locale comme indicateur simple.

---

### `cloud_cover`

La couverture nuageuse influence directement la quantité de rayonnement solaire atteignant la surface.

En journée :

```
moins de nuages
      ↓
davantage de rayonnement solaire
      ↓
potentiellement davantage de chauffage
```

Elle constitue donc une variable naturelle à tester.

---

### `shortwave_radiation`

Le rayonnement solaire reçu par la surface est directement relié au bilan énergétique.

Il devrait donc contenir une information intéressante pour la température.

Il faudra cependant faire attention à ne pas utiliser une valeur du futur qui ne serait pas connue au moment de la prédiction.

---

### `wind_speed_10m`

Le vent permet le transport de masses d’air.

Une modification de la vitesse du vent peut être associée à l’arrivée d’un nouveau régime météorologique.

---

### `wind_direction_10m`

La direction est importante car un vent provenant d’une région chaude n’a pas le même effet qu’un vent provenant d’une région plus fraîche.

Cependant, la direction du vent ne doit pas être utilisée directement comme une variable numérique classique.

Par exemple :

```
359°
```

et

```
1°
```

correspondent presque à la même direction, alors qu’une régression linéaire les considérerait comme très éloignées.

Nous transformerons donc cette variable avec :

\[ wind\_sin = \sin(\theta) \]\[ wind\_cos = \cos(\theta) \]

---

### `precipitation`

La pluie peut modifier l’état du sol et les échanges énergétiques avec l’atmosphère.

Elle permet également de caractériser le type de situation météorologique observée les jours précédents.

Nous la conservons donc dans le premier dataset.

---

### `soil_moisture_0_to_7cm`

L’humidité du sol peut jouer un rôle dans les épisodes de chaleur prolongée.

Un sol humide peut utiliser une partie de l’énergie reçue pour l’évaporation, tandis qu’un sol très sec peut davantage favoriser le chauffage de la surface.

Cette variable est cependant moins prioritaire pour notre première régression.

Nous la collectons afin de pouvoir la tester ultérieurement.

---

# 8. Pourquoi ne pas prendre toutes les variables Open-Meteo ?

Notre but n’est pas de donner immédiatement au modèle tout ce qui est disponible.

Nous voulons comprendre ce que chaque groupe de variables apporte.

Certaines variables disponibles sont également construites à partir d’autres variables.

Par exemple, une température ressentie peut déjà combiner :

- température ;
- humidité ;
- vent ;
- rayonnement.

Utiliser simultanément les variables originales et une variable déjà dérivée d’elles peut rendre l’interprétation plus difficile.

Notre logique sera donc :

```
variables physiques simples
        ↓
premier modèle
        ↓
évaluation
        ↓
ajout ciblé de nouvelles variables
```

---

# 9. Pourquoi télécharger des données horaires ?

Même si notre première cible est journalière, nous choisissons de récupérer les données à une fréquence **horaire**.

Cela nous donne plus de liberté.

À partir des observations horaires, nous pourrons calculer nous-mêmes :

```
température maximale quotidienne
température minimale quotidienne
température moyenne quotidienne
humidité moyenne
pression moyenne
rayonnement total
précipitations cumulées
vent moyen
vent maximal
etc.
```

Si nous téléchargions uniquement des valeurs déjà agrégées à la journée, nous perdrions définitivement la structure intra-journalière.

Le coût de stockage étant faible pour une seule localisation, conserver les données horaires nous paraît plus intéressant.

---

# 10. Passage des données horaires au dataset de Machine Learning

Nous partirons donc d’un dataset horaire de la forme :

```
timestamp
temperature_2m
relative_humidity_2m
dew_point_2m
pressure_msl
cloud_cover
shortwave_radiation
wind_speed_10m
wind_direction_10m
precipitation
soil_moisture_0_to_7cm
```

Puis nous construirons un dataset quotidien.

Une ligne représentera un jour.

Exemple conceptuel :

|jour|Tmax|Tmin|Tmean|humidité|pression|rayonnement|vent|précipitations|
|---|---|---|---|---|---|---|---|---|
|Jour 1|...|...|...|...|...|...|...|...|
|Jour 2|...|...|...|...|...|...|...|...|

La cible sera décalée d’une journée :

```
Données du jour t  ────────>  Tmax du jour t+1
```

---

# 11. Ajouter la mémoire du passé : les variables retardées

La météo est un processus temporel.

Utiliser uniquement les valeurs du jour \(t\) risque donc de perdre de l’information présente dans les jours précédents.

Nous introduirons progressivement des variables comme :

```
Tmax(t)
Tmax(t-1)
Tmax(t-2)
Tmax(t-3)
```

et éventuellement :

```
pressure(t)
pressure(t-1)

humidity(t)
humidity(t-1)

...
```

Le nombre de jours à conserver ne sera pas choisi arbitrairement une fois pour toutes.

Nous testerons plusieurs fenêtres temporelles et comparerons les performances.

---

# 12. Représenter la saison

La température dépend fortement de la période de l’année.

Une simple régression doit donc recevoir une information permettant de comprendre qu’un jour de janvier et un jour de juillet n’ont pas le même comportement moyen.

Nous pouvons créer :

```
day_of_year
```

mais une représentation cyclique est préférable.

Nous utiliserons :

\[ day\_sin = \sin\left( 2\pi \frac{day\_of\_year}{365.25} \right) \]\[ day\_cos = \cos\left( 2\pi \frac{day\_of\_year}{365.25} \right) \]

Cela permet au modèle de comprendre que :

```
31 décembre
```

et

```
1er janvier
```

sont très proches dans le cycle annuel.

---

# 13. Point critique : éviter la fuite de données

Pour prédire la température maximale de demain, nous ne devons utiliser que des informations réellement disponibles aujourd’hui.

Par exemple, ceci serait incorrect :

```
humidité réelle de demain
pression réelle de demain
rayonnement réel de demain
        ↓
prédiction de la température de demain
```

Le modèle utiliserait alors des informations provenant du futur.

La règle que nous devons toujours appliquer est :

> **Pour chaque variable, demander : est-ce que cette information serait réellement disponible au moment où la prédiction est effectuée ?**

Dans notre première approche, nous utiliserons donc principalement les observations historiques jusqu’au jour \(t\) pour prédire le jour \(t+1\).

---

# 14. Première série d’expériences

Nous ne voulons pas entraîner directement un seul gros modèle.

Nous allons construire une progression.

## Expérience 0 — baseline naïve

\[ \hat{T}_{\max}(t+1) = T_{\max}(t) \]

Objectif :

> mesurer ce que donne une simple hypothèse de persistance.

---

## Expérience 1 — température seulement

Régression linéaire utilisant par exemple :

```
Tmax(t)
Tmin(t)
Tmean(t)
day_sin
day_cos
```

Objectif :

> vérifier ce qu’apportent l’historique thermique et la saisonnalité.

---

## Expérience 2 — ajout des autres variables météorologiques

Ajouter progressivement :

```
humidité
point de rosée
pression
couverture nuageuse
rayonnement solaire
vent
précipitations
```

Objectif :

> déterminer si l’état plus complet de l’atmosphère apporte une amélioration par rapport à la température seule.

---

## Expérience 3 — ajout d’une mémoire temporelle plus longue

Ajouter plusieurs jours précédents :

```
t
t-1
t-2
...
```

Objectif :

> déterminer combien de passé est réellement utile à une régression linéaire.

---

# 15. Comment décider si une variable est utile ?

Nous ne voulons pas conclure qu’une variable est intéressante uniquement parce qu’elle semble physiquement pertinente ou parce qu’elle est corrélée à la température.

Nous suivrons trois niveaux de raisonnement :

```
1. Raisonnement physique
   ↓
La variable pourrait-elle influencer la température ?

2. Analyse statistique
   ↓
Observe-t-on une relation dans les données ?

3. Validation du modèle
   ↓
L'ajout de la variable améliore-t-il réellement les prédictions
sur des données jamais vues ?
```

La troisième question sera décisive.

Nous pourrons réaliser des **ablation studies** :

```
modèle avec pression
vs
modèle sans pression
```

ou :

```
modèle avec vent
vs
modèle sans vent
```

Cela nous permettra de comprendre concrètement l’utilité de chaque information.

---

# 16. Évaluation

Nous commencerons avec des métriques simples de régression, notamment :

- MAE ;
- RMSE.

Mais il faudra progressivement distinguer deux performances.

## Performance générale

Le modèle prédit-il correctement les températures en moyenne ?

## Performance pendant les périodes chaudes

Le modèle reste-t-il bon précisément lorsque les températures deviennent extrêmes ?

C’est particulièrement important car notre objectif final concerne les vagues de chaleur.

Un modèle peut être très bon en moyenne mais sous-estimer systématiquement les températures extrêmes.

Nous pourrons donc calculer plus tard :

```
MAE globale
MAE pendant les journées très chaudes
MAE pendant les épisodes de vague de chaleur
```

---

# 17. Ce que nous attendons de cette première phase

Nous ne cherchons pas encore à obtenir un modèle de prévision météorologique de haut niveau.

Cette première phase doit surtout répondre à plusieurs questions.

### Question 1

Une régression linéaire locale peut-elle battre la baseline :

```
Tmax demain = Tmax aujourd'hui
```

### Question 2

Quelles variables météorologiques locales apportent réellement de l’information ?

### Question 3

Combien de jours d’historique sont utiles ?

### Question 4

Le modèle se comporte-t-il correctement pendant les températures extrêmes ?

### Question 5

Quelles sont les limites d’une approche utilisant uniquement un point géographique ?

---

# 18. Ce que nous ferons ensuite

Une fois cette première baseline bien comprise, nous pourrons ajouter de la complexité de manière contrôlée.

## Étape suivante possible : contexte spatial

La météo n’est pas purement locale.

Nous pourrons récupérer les mêmes variables autour de la ville cible :

```
      x   x   x
      x   X   x
      x   x   x
```

où `X` représente la ville cible.

Nous pourrons alors comparer :

```
modèle local
vs
modèle avec zone environnante
```

et mesurer précisément le gain.

---

## Ensuite : horizons de prédiction plus longs

Une fois la prédiction à \(t+1\) maîtrisée :

```
demain
```

nous pourrons essayer :

```
t+2
t+3
...
```

et étudier comment la performance se dégrade avec l’horizon.

---

## Ensuite : aller vers la véritable vague de chaleur

Lorsque nous disposerons de bonnes prédictions de température, nous pourrons formaliser la notion de vague de chaleur :

- seuils locaux ;
- températures maximales ;
- températures minimales nocturnes ;
- durée minimale ;
- intensité.

Nous pourrons alors :

1. dériver des alertes à partir des températures prédites ;
2. ou construire directement un modèle de classification de vague de chaleur.

---

## Ensuite : modèles plus complexes

Seulement une fois les limites de la régression linéaire comprises, nous pourrons comparer avec :

```
Random Forest
Gradient Boosting / XGBoost
réseaux récurrents
modèles temporels
modèles spatio-temporels
Transformers
etc.
```

L’objectif sera alors de pouvoir répondre à la question :

> **Quelle complexité supplémentaire apporte réellement une amélioration ?**

et non simplement :

> **Quel est le modèle le plus sophistiqué que nous pouvons entraîner ?**

---

# 19. Fil conducteur global du projet

Notre démarche peut être résumée ainsi :

```
Définir clairement la cible
        │
        ▼
Tmax du lendemain pour une localisation
        │
        ▼
Récupérer les données Open-Meteo horaires
        │
        ▼
Construire des variables journalières
        │
        ▼
Créer une baseline naïve
        │
        ▼
Régression linéaire avec température
        │
        ▼
Ajouter saisonnalité
        │
        ▼
Ajouter variables météorologiques
        │
        ▼
Ajouter historique sur plusieurs jours
        │
        ▼
Comparer les performances
        │
        ▼
Analyser particulièrement les jours chauds
        │
        ▼
Identifier les limites du modèle local
        │
        ▼
Ajouter du contexte spatial
        │
        ▼
Tester des horizons de prédiction plus longs
        │
        ▼
Formaliser la détection de vagues de chaleur
        │
        ▼
Comparer avec des modèles plus complexes
```

---

# 20. Principe méthodologique à conserver

La règle principale pour la suite du projet est :

> **Ne pas ajouter de complexité sans savoir quelle question cette complexité est censée résoudre.**

À chaque nouvelle étape, nous voulons pouvoir dire :

```
voici notre modèle actuel
voici sa limite
voici l'information que nous ajoutons
voici pourquoi elle pourrait aider
voici l'expérience qui permet de vérifier si elle aide réellement
```

C’est ce fil conducteur qui doit permettre au projet de rester compréhensible, scientifique et progressif.