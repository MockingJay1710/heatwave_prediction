# Révision Machine Learning — Kaggle Intro, Intermediate & Feature Engineering

Ce document regroupe dans un ordre logique les notions principales vues dans les trois cours Kaggle :

1. **Intro to Machine Learning**
2. **Intermediate Machine Learning**
3. **Feature Engineering**

L’objectif n’est pas simplement de résumer les notebooks Kaggle, mais de reconstruire une vision cohérente du workflow Machine Learning :

\[
\text{données brutes}
\rightarrow
\text{représentation}
\rightarrow
\text{préprocessing}
\rightarrow
\text{modèle}
\rightarrow
\text{évaluation}
\rightarrow
\text{itération}
\rightarrow
\text{production}
\]

---

# 1. Le problème fondamental du Machine Learning supervisé

En apprentissage supervisé, on possède des exemples :

\[
(x_1,y_1), (x_2,y_2), \dots, (x_n,y_n)
\]

où :

- \(x_i\) contient les **features** de l'observation ;
- \(y_i\) contient la **target** que l'on veut prédire.

On cherche à apprendre une fonction :

\[
f : X \rightarrow y
\]

de telle sorte que :

\[
\hat y = f(x)
\]

soit proche du vrai \(y\), y compris pour des observations que le modèle n'a jamais vues.

C'est le concept de **généralisation**.

---

## 1.1 Features et target

Exemple : prédiction du prix d'une maison.

Features :

\[
X =
\begin{bmatrix}
\text{surface} &
\text{nombre de chambres} &
\text{année de construction} &
\text{quartier}
\end{bmatrix}
\]

Target :

\[
y = \text{prix}
\]

Dans Pandas :

```python
y = data["Price"]

features = [
    "Rooms",
    "Bathroom",
    "Landsize",
    "BuildingArea",
    "YearBuilt"
]

X = data[features]
```

Chaque **ligne** correspond à une observation.

Chaque **colonne** correspond à une feature.

---

# 2. Régression et classification

Deux grands types de problèmes apparaissent souvent.

## Régression

La target est continue :

\[
y \in \mathbb{R}
\]

Exemples :

- prix d'une maison ;
- température ;
- demande énergétique ;
- durée d'une panne.

---

## Classification

La target appartient à un ensemble de classes.

Exemple binaire :

\[
y \in \{0,1\}
\]

ou :

\[
y \in \{\text{spam}, \text{non-spam}\}
\]

---

# 3. Entraînement et inférence

Avec scikit-learn :

```python
model.fit(X_train, y_train)
```

signifie :

> apprendre les paramètres du modèle à partir des données d'entraînement.

On peut représenter cela par :

\[
(X_{\text{train}}, y_{\text{train}})
\rightarrow
\hat f
\]

Puis :

```python
predictions = model.predict(X_new)
```

correspond à l'inférence :

\[
X_{\text{new}}
\xrightarrow{\hat f}
\hat y
\]

Il faut donc distinguer :

\[
\boxed{\text{training} \neq \text{inference}}
\]

---

# 4. Premier modèle : Decision Tree

Un arbre de décision construit des règles successives.

Exemple :

```text
Surface < 100 m² ?
├── oui
│   └── quartier ?
│       ├── centre → 250 k€
│       └── périphérie → 180 k€
└── non
    └── chambres < 4 ?
        ├── oui → 320 k€
        └── non → 430 k€
```

Le modèle découpe l'espace des observations en plusieurs régions.

Dans chaque feuille, il produit une prédiction.

---

## 4.1 Comment choisit-il les séparations ?

Il cherche des règles permettant de créer des groupes aussi homogènes que possible vis-à-vis de la target.

Exemple :

| Surface | Prix |
|---:|---:|
| 50 | 150k |
| 60 | 160k |
| 70 | 170k |
| 140 | 350k |
| 150 | 370k |

La règle :

\[
Surface < 100
\]

sépare naturellement :

```text
150k, 160k, 170k
```

de :

```text
350k, 370k
```

---

# 5. Pourquoi on ne mesure pas la performance sur le train

Supposons :

```python
model.fit(X, y)
predictions = model.predict(X)
```

Puis on mesure l'erreur entre :

\[
y
\]

et :

\[
\hat y
\]

Le résultat peut être excellent.

Mais ce n'est pas une bonne mesure de généralisation, car le modèle est évalué sur les données qu'il vient d'utiliser pour apprendre.

Analogie :

> un étudiant mémorise un examen puis repasse exactement le même examen.

Il peut obtenir 20/20 sans réellement savoir résoudre de nouveaux problèmes.

---

# 6. Train / Validation / Test

On sépare généralement les données en trois rôles.

## Training set

Utilisé pour apprendre les paramètres :

\[
D_{\text{train}}
\]

```python
model.fit(X_train, y_train)
```

---

## Validation set

Utilisé pour :

- comparer des modèles ;
- choisir des hyperparamètres ;
- mesurer la généralisation pendant le développement.

\[
D_{\text{val}}
\]

---

## Test set

Utilisé à la fin pour obtenir une estimation finale aussi indépendante que possible :

\[
D_{\text{test}}
\]

---

## 6.1 Train/validation split

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    random_state=1
)
```

Puis :

```python
model.fit(X_train, y_train)

predictions = model.predict(X_valid)
```

Le principe fondamental est :

\[
D_{\text{validation}}
\notin
D_{\text{training}}
\]

---

# 7. Fonction de perte et métriques

Pour mesurer les erreurs, on utilise une **loss function** ou une métrique.

Dans le cours Kaggle de régression, on utilise notamment la **Mean Absolute Error**.

\[
MAE
=
\frac{1}{n}
\sum_{i=1}^{n}
|y_i-\hat y_i|
\]

Exemple :

\[
y=[200,300,400]
\]

\[
\hat y=[220,270,390]
\]

Erreurs absolues :

\[
20,\;30,\;10
\]

Donc :

\[
MAE=\frac{20+30+10}{3}=20
\]

En sklearn :

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_valid, predictions)
```

---

# 8. Généralisation, underfitting et overfitting

Un bon modèle ne doit pas seulement bien apprendre le train.

Il doit généraliser.

---

## 8.1 Underfitting

Le modèle est trop simple pour représenter la structure des données.

Exemple :

```text
Surface < 100 ?
├── oui → 180k
└── non → 350k
```

Il ignore beaucoup d'informations importantes.

Conséquence :

- erreur élevée sur le train ;
- erreur élevée sur la validation.

---

## 8.2 Overfitting

Le modèle est trop complexe et apprend des détails spécifiques au dataset.

Il peut pratiquement mémoriser les observations.

Conséquence :

- erreur très faible sur le train ;
- erreur élevée sur la validation.

---

## 8.3 Compromis

Schématiquement :

```text
Erreur
  ↑
  │            validation
  │          \          /
  │           \________/
  │
  │\
  │ \
  │  \________________ train
  │
  └────────────────────────→ complexité
       underfit       overfit
```

On cherche le niveau de complexité produisant la meilleure généralisation.

---

# 9. Paramètres et hyperparamètres

## Paramètres

Appris automatiquement à partir des données.

Par exemple dans un arbre :

- variables choisies pour les splits ;
- seuils de séparation ;
- prédictions dans les feuilles.

Ils sont appris lors de :

```python
model.fit(...)
```

---

## Hyperparamètres

Choisis avant l'entraînement.

Exemple :

```python
DecisionTreeRegressor(
    max_leaf_nodes=100
)
```

`max_leaf_nodes` est un hyperparamètre.

Autres exemples :

- `max_depth`
- `n_estimators`
- `learning_rate`
- `batch_size`
- nombre de couches d'un réseau neuronal

---

# 10. Random Forest

Un arbre seul peut être instable.

Une Random Forest entraîne plusieurs arbres différents puis combine leurs prédictions.

Pour une régression :

\[
\hat y
=
\frac{1}{M}
\sum_{m=1}^{M}T_m(x)
\]

où \(T_m\) est le \(m\)-ième arbre.

---

## 10.1 Intuition

Supposons :

\[
T_1(x)=310
\]

\[
T_2(x)=330
\]

\[
T_3(x)=295
\]

\[
T_4(x)=320
\]

Alors :

\[
\hat y
=
\frac{310+330+295+320}{4}
\approx314
\]

Les erreurs différentes des arbres ont tendance à partiellement se compenser.

---

## 10.2 Bagging

La Random Forest repose sur l'idée de **bagging** :

\[
\boxed{
\text{beaucoup de modèles différents}
\rightarrow
\text{prédiction plus stable}
}
\]

Chaque arbre voit des variations des données et des sous-ensembles de features.

Cela réduit la corrélation entre les arbres et leur variance collective.

---

# 11. Le vrai problème : les données réelles sont imparfaites

Dans le premier cours, les données sont relativement propres.

Dans la réalité, on rencontre :

```text
NaN
catégories
dates
chaînes de caractères
colonnes inutiles
forte cardinalité
outliers
préprocessing complexe
```

Exemple :

| Surface | Quartier | Garage | Année | Prix |
|---:|---|---|---:|---:|
| 120 | Centre | yes | 2001 | 350k |
| NaN | Nord | no | 1995 | 270k |
| 90 | Centre | NaN | NaN | 290k |
| 180 | Sud | yes | 2015 | 520k |

Avant le modèle, il faut donc construire un **préprocessing reproductible**.

---

# 12. Missing Values

Les valeurs manquantes peuvent être traitées de plusieurs façons.

---

## 12.1 Supprimer la colonne

```python
cols_with_missing = [
    col for col in X_train.columns
    if X_train[col].isnull().any()
]

X_train_reduced = X_train.drop(cols_with_missing, axis=1)
```

Avantage :

- simple.

Inconvénient :

- on peut perdre beaucoup d'information.

---

# 13. Imputation

On remplace les valeurs manquantes par une valeur calculée sur le train.

Exemple :

\[
[100,120,?,90,110]
\]

Moyenne :

\[
\mu =105
\]

Après imputation :

\[
[100,120,105,90,110]
\]

Avec sklearn :

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

X_train_imputed = imputer.fit_transform(X_train)
X_valid_imputed = imputer.transform(X_valid)
```

---

# 14. `fit`, `transform`, `fit_transform`

C'est une distinction fondamentale.

## `fit`

Apprend des paramètres à partir des données.

Exemple :

```python
imputer.fit(X_train)
```

peut apprendre :

\[
\mu_{\text{surface}} = 125.7
\]

---

## `transform`

Applique les paramètres déjà appris.

```python
imputer.transform(X_valid)
```

---

## `fit_transform`

Équivaut conceptuellement à :

```python
transformer.fit(X)
transformer.transform(X)
```

Donc typiquement :

```python
X_train = transformer.fit_transform(X_train)
X_valid = transformer.transform(X_valid)
```

et non :

```python
transformer.fit_transform(X_valid)
```

---

# 15. Missingness comme information

Une valeur manquante peut elle-même contenir de l'information.

Exemple :

```text
GarageArea = NaN
```

peut signifier :

> la maison n'a pas de garage.

On peut donc créer :

```text
GarageArea_missing
```

avec :

\[
0 = \text{présent}
\]

\[
1 = \text{manquant}
\]

Puis imputer la valeur.

Ainsi le modèle conserve l'information :

> cette observation avait une valeur manquante.

---

# 16. Variables catégorielles

Exemple :

```text
Neighborhood:
- Centre
- North
- South
- West
```

Beaucoup de modèles attendent des entrées numériques.

Il faut donc encoder les catégories.

---

# 17. Ordinal Encoding

Si les catégories possèdent un ordre naturel :

```text
Poor
Average
Good
Excellent
```

on peut encoder :

\[
Poor\rightarrow0
\]

\[
Average\rightarrow1
\]

\[
Good\rightarrow2
\]

\[
Excellent\rightarrow3
\]

Avec :

```python
from sklearn.preprocessing import OrdinalEncoder
```

---

## 17.1 Limite

Pour :

```text
Paris
Nantes
Lyon
```

encoder :

\[
Paris=0,\ Nantes=1,\ Lyon=2
\]

introduit artificiellement :

\[
Paris < Nantes < Lyon
\]

et des distances numériques qui n'ont aucun sens.

---

# 18. One-Hot Encoding

On transforme :

| City |
|---|
| Paris |
| Lyon |
| Nantes |

en :

| Paris | Lyon | Nantes |
|---:|---:|---:|
| 1 | 0 | 0 |
| 0 | 1 | 0 |
| 0 | 0 | 1 |

Avec sklearn :

```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    handle_unknown="ignore"
)
```

`handle_unknown="ignore"` permet d'éviter une erreur si une nouvelle catégorie apparaît au moment de l'inférence.

---

# 19. Cardinalité

La **cardinalité** correspond au nombre de valeurs distinctes.

Une feature :

```text
country
```

peut avoir 20 catégories.

Une feature :

```text
user_id
```

peut avoir 1 000 000 de catégories.

Avec One-Hot Encoding :

\[
1\,000\,000\text{ catégories}
\rightarrow
1\,000\,000\text{ colonnes potentielles}
\]

Ce n'est généralement pas souhaitable.

---

# 20. Pipelines

Un vrai modèle comprend rarement seulement :

```text
model
```

Il comprend plutôt :

```text
raw data
   ↓
imputation
   ↓
encoding
   ↓
feature engineering
   ↓
scaling éventuel
   ↓
model
```

Le pipeline encapsule toute cette logique.

Mathématiquement :

\[
F
=
M \circ P_k \circ \dots \circ P_2 \circ P_1
\]

où :

- \(P_i\) sont les transformations ;
- \(M\) est le modèle.

---

## 20.1 Exemple sklearn

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)
```

Puis :

```python
pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_valid)
```

---

# 21. ColumnTransformer

Toutes les colonnes n'ont pas besoin du même traitement.

Exemple :

```text
Age      → numérique
Salary   → numérique
City     → catégorielle
Job      → catégorielle
```

On peut utiliser :

```python
from sklearn.compose import ColumnTransformer
```

---

## 21.1 Pipeline complet

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

numerical_transformer = SimpleImputer(
    strategy="median"
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_cols
        ),
        (
            "cat",
            categorical_transformer,
            categorical_cols
        )
    ]
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_valid)
```

---

# 22. Pourquoi les pipelines sont importants

Ils permettent :

- la reproductibilité ;
- d'éviter d'oublier une transformation ;
- d'éviter certaines fuites de données ;
- d'appliquer exactement les mêmes transformations au train et à l'inférence ;
- de faciliter le déploiement.

Conceptuellement :

\[
\boxed{
\text{le preprocessing fait partie du modèle déployé}
}
\]

Le système réellement utilisé en production est plutôt :

\[
\boxed{
\text{preprocessing}
+
\text{feature engineering}
+
\text{model}
}
\]

---

# 23. Cross-Validation

Un simple train/validation split peut produire une mesure instable.

Exemple :

```text
split 1 → MAE = 20
split 2 → MAE = 27
split 3 → MAE = 22
```

Cela dépend du hasard du split.

La **K-Fold Cross-Validation** utilise plusieurs partitions.

---

## 23.1 Exemple à 5 folds

On sépare :

```text
A B C D E
```

Puis :

```text
Run 1:
TRAIN = B C D E
VALID = A

Run 2:
TRAIN = A C D E
VALID = B

Run 3:
TRAIN = A B D E
VALID = C

...
```

On obtient :

\[
Score_1,\dots,Score_5
\]

Puis :

\[
CV
=
\frac{1}{5}
\sum_{k=1}^{5}Score_k
\]

---

# 24. Pipeline + Cross-Validation

Supposons qu'on fasse :

```python
X = imputer.fit_transform(X)

cross_val_score(model, X, y)
```

L'imputer a appris sur tout \(X\), donc aussi indirectement sur les futures folds de validation.

C'est une fuite d'information.

Il vaut mieux faire :

```python
pipeline = Pipeline([
    ("imputer", SimpleImputer()),
    ("model", RandomForestRegressor())
])
```

puis :

```python
cross_val_score(
    pipeline,
    X,
    y,
    cv=5
)
```

À chaque fold :

```text
train fold
   ↓
fit preprocessing
   ↓
fit model

validation fold
   ↓
transform avec paramètres du train
   ↓
predict
```

---

# 25. Data Leakage

Une fuite de données apparaît lorsqu'une information qui ne devrait pas être disponible au moment réel de la prédiction influence le modèle.

C'est particulièrement dangereux parce que :

```text
le code fonctionne
+
les scores semblent excellents
```

alors que la performance réelle s'effondre.

---

# 26. Target Leakage

Supposons que l'on prédise :

\[
Y=\text{patient malade ?}
\]

Features :

```text
Age
Weight
BloodPressure
TreatmentReceived
```

Mais si `TreatmentReceived` est décidé après le diagnostic :

\[
Disease \rightarrow Treatment
\]

Le modèle utilise une information du futur.

Cela ne correspond pas au vrai scénario de prédiction.

---

## 26.1 Autre exemple

On veut prédire :

\[
\text{default de crédit}
\]

et on utilise :

```text
AmountRecoveredAfterDefault
```

Cette information n'existe qu'après le défaut.

Elle révèle pratiquement la target.

---

# 27. Train-Test Contamination

Supposons :

```python
imputer.fit(X)
```

avant :

```python
train_test_split(X, y)
```

Alors les statistiques de validation ont influencé l'imputer.

Même problème avec :

```python
scaler.fit(X)
encoder.fit(X)
feature_selector.fit(X, y)
```

La règle générale est :

\[
\boxed{
\text{split avant tout apprentissage du preprocessing}
}
\]

---

# 28. Feature Engineering

Le feature engineering consiste à améliorer la **représentation du problème**.

On applique une transformation :

\[
z=\phi(x)
\]

puis le modèle apprend :

\[
\hat y=f(z)=f(\phi(x))
\]

L'objectif est de créer un espace dans lequel la relation avec la target est plus facile à apprendre.

---

# 29. Preprocessing vs Feature Engineering

Une distinction utile :

## Preprocessing

Rendre les données exploitables.

Exemples :

- gérer les NaN ;
- encoder les catégories ;
- standardiser.

---

## Feature Engineering

Créer une représentation plus pertinente.

Exemples :

\[
BMI = \frac{Weight}{Height^2}
\]

\[
PricePerSquareMeter
=
\frac{Price}{Area}
\]

\[
PowerToWeight
=
\frac{Horsepower}{Weight}
\]

En pratique, la frontière n'est pas toujours parfaitement nette.

---

# 30. Pourquoi le Feature Engineering fonctionne

Supposons que :

\[
y=\frac{x_1}{x_2}
\]

Le modèle reçoit seulement :

\[
x_1,x_2
\]

Il doit apprendre la division.

Mais si on crée :

\[
z=\frac{x_1}{x_2}
\]

alors :

\[
y=z
\]

Le problème devient beaucoup plus simple.

C'est l'idée générale :

\[
\boxed{
\text{une bonne représentation peut simplifier énormément la fonction à apprendre}
}
\]

---

# 31. Mutual Information

La Mutual Information sert à mesurer la dépendance statistique entre une feature \(X\) et la target \(Y\).

Intuition :

> combien connaître \(X\) réduit-il notre incertitude sur \(Y\) ?

---

## 31.1 Entropie

Pour une variable discrète :

\[
H(Y)
=
-\sum_y
P(Y=y)\log P(Y=y)
\]

L'entropie mesure l'incertitude.

---

## 31.2 Mutual Information

\[
\boxed{
I(X;Y)
=
H(Y)-H(Y|X)
}
\]

Si \(X\) n'apporte aucune information :

\[
H(Y|X)=H(Y)
\]

donc :

\[
I(X;Y)=0
\]

---

## 31.3 Autre expression

\[
I(X;Y)
=
\int\int
p(x,y)
\log
\frac{p(x,y)}
{p(x)p(y)}
dxdy
\]

Si \(X\) et \(Y\) sont indépendants :

\[
p(x,y)=p(x)p(y)
\]

alors la MI vaut 0.

---

# 32. Mutual Information vs corrélation

La corrélation classique mesure surtout une relation linéaire.

Exemple :

\[
Y=X^2
\]

avec un \(X\) symétrique autour de zéro.

La corrélation linéaire peut être proche de 0 alors que \(X\) détermine parfaitement \(Y\).

La Mutual Information peut détecter une dépendance non linéaire.

Avec sklearn :

```python
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X, y)
```

ou :

```python
from sklearn.feature_selection import mutual_info_classif
```

---

## 32.1 Limites de la MI

Une MI élevée ne signifie pas causalité.

\[
I(X;Y)>0
\]

signifie seulement :

> \(X\) et \(Y\) possèdent une dépendance statistique.

De plus, une analyse feature par feature peut manquer des interactions.

Exemple XOR :

\[
Y=X_1\oplus X_2
\]

Chaque feature seule peut sembler peu informative, alors que leur combinaison détermine parfaitement \(Y\).

---

# 33. Mathematical Transformations

On peut créer des features à partir d'opérations mathématiques.

Exemples :

\[
ratio=\frac{x_1}{x_2}
\]

\[
interaction=x_1x_2
\]

\[
difference=x_1-x_2
\]

\[
sum=x_1+x_2
\]

Exemple :

```python
autos["stroke_ratio"] = (
    autos["stroke"] / autos["bore"]
)
```

---

# 34. Interactions

Dans un modèle linéaire :

\[
y
=
\beta_0
+
\beta_1x_1
+
\beta_2x_2
\]

l'effet de \(x_1\) est :

\[
\frac{\partial y}{\partial x_1}
=
\beta_1
\]

Il ne dépend pas de \(x_2\).

Avec une interaction :

\[
y
=
\beta_0
+
\beta_1x_1
+
\beta_2x_2
+
\beta_3x_1x_2
\]

on obtient :

\[
\frac{\partial y}{\partial x_1}
=
\beta_1+\beta_3x_2
\]

L'effet de \(x_1\) dépend maintenant de \(x_2\).

---

# 35. Transformations logarithmiques

Pour une distribution très asymétrique :

```text
20
30
40
50
10 000
```

on peut utiliser :

\[
x'=\log(1+x)
\]

En Python :

```python
import numpy as np

df["LogIncome"] = np.log1p(df["Income"])
```

Cela compresse les très grandes valeurs.

---

# 36. Count Features

Plusieurs indicateurs binaires :

```text
HasGarage
HasPool
HasGarden
HasBalcony
HasBasement
```

peuvent être combinés en :

\[
AmenitiesCount
=
HasGarage
+
HasPool
+
HasGarden
+
HasBalcony
+
HasBasement
\]

```python
df["AmenitiesCount"] = df[features].sum(axis=1)
```

Cela crée un concept plus global :

> nombre total d'équipements.

---

# 37. Décomposer une feature

Une seule colonne peut contenir plusieurs informations.

Exemple :

```text
2026-09-22 08:32:14
```

peut être décomposée en :

```text
year
month
day
hour
weekday
```

```python
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.dayofweek
df["hour"] = df["date"].dt.hour
```

---

## 37.1 Autres exemples

Une adresse :

```text
12 Rue de Paris, Nantes, 44000
```

peut contenir :

```text
street
city
postal_code
```

Même idée pour :

- URL ;
- SKU ;
- identifiant produit ;
- numéro de version ;
- code géographique.

---

# 38. Combiner des features

On peut aussi faire l'inverse.

Exemple :

```text
brand = BMW
body_style = SUV
```

Créer :

```text
BMW_SUV
```

Avec :

```python
df["brand_style"] = (
    df["brand"] + "_" + df["body_style"]
)
```

Cela matérialise une interaction entre deux catégories.

---

# 39. Group Transforms

On peut calculer une statistique de groupe.

Exemple :

| Personne | Ville | Salaire |
|---|---|---:|
| A | Nantes | 40 |
| B | Nantes | 50 |
| C | Paris | 70 |
| D | Paris | 90 |

Moyenne par ville :

\[
\mu_{\text{Nantes}}=45
\]

\[
\mu_{\text{Paris}}=80
\]

En Pandas :

```python
df["AverageSalaryCity"] = (
    df.groupby("city")["salary"]
      .transform("mean")
)
```

On peut ensuite créer :

\[
RelativeSalary
=
\frac{Salary}{AverageSalaryCity}
\]

Cela donne simultanément :

- l'information individuelle ;
- le contexte du groupe.

---

# 40. Attention au leakage dans le Feature Engineering

Une statistique calculée sur tout le dataset peut utiliser indirectement la validation.

Exemple :

```python
group_means = (
    full_dataset
    .groupby("city")["salary"]
    .mean()
)
```

Si la statistique est ensuite utilisée comme feature, les données de validation ont influencé le train.

Le principe reste :

\[
\boxed{
fit\ sur\ train
\rightarrow
transform\ train/validation/test
}
\]

---

# 41. K-Means comme Feature Engineering

K-Means est un algorithme de clustering non supervisé.

Mais il peut être utilisé pour créer une nouvelle feature :

```text
cluster_id
```

Exemple avec latitude/longitude :

```text
zone 0
zone 1
zone 2
```

Cette feature peut aider un modèle supervisé.

---

# 42. Fonctionnement de K-Means

On choisit :

\[
K
\]

centroïdes :

\[
\mu_1,\dots,\mu_K
\]

Chaque observation \(x_i\) est affectée au centroïde le plus proche :

\[
c_i
=
\arg\min_k
\|x_i-\mu_k\|^2
\]

Puis les centroïdes sont recalculés :

\[
\mu_k
=
\frac{1}{|C_k|}
\sum_{x_i\in C_k}x_i
\]

On répète jusqu'à convergence.

---

## 42.1 Objectif

K-Means minimise :

\[
\boxed{
J
=
\sum_{k=1}^{K}
\sum_{x_i\in C_k}
\|x_i-\mu_k\|^2
}
\]

Il cherche donc des clusters compacts.

---

# 43. Pourquoi un cluster peut aider le modèle

Supposons :

```text
zone A:
price ≈ 3000 × surface

zone B:
price ≈ 5000 × surface

zone C:
price ≈ 9000 × surface
```

Sans cluster :

\[
Price=f(Surface,Latitude,Longitude)
\]

Avec cluster :

\[
Price=f(Surface,Cluster)
\]

La nouvelle feature résume une structure spatiale complexe.

---

# 44. Scaling et méthodes basées sur la distance

K-Means utilise la distance euclidienne :

\[
d(x,y)
=
\sqrt{
\sum_j(x_j-y_j)^2
}
\]

Supposons :

```text
Age    : 20 → 80
Salary : 20 000 → 200 000
```

La feature `Salary` dominera la distance simplement à cause de son unité.

On standardise souvent :

\[
z
=
\frac{x-\mu}{\sigma}
\]

avant K-Means.

---

# 45. PCA — Principal Component Analysis

PCA cherche de nouvelles directions représentant le maximum de variance des données.

Elle est non supervisée :

\[
PCA(X)
\]

ne regarde pas \(y\).

---

# 46. Intuition géométrique

Supposons deux features fortement corrélées :

```text
x2
↑
│           ●
│        ●
│      ●
│    ●
│  ●
└────────────→ x1
```

Les points varient surtout le long d'une diagonale.

PCA cherche un axe aligné avec cette direction.

---

# 47. Première composante principale

On cherche :

\[
w_1
\]

tel que :

\[
z_1 = Xw_1
\]

ait une variance maximale.

\[
\boxed{
w_1
=
\arg\max_{\|w\|=1}
Var(Xw)
}
\]

La deuxième composante doit être orthogonale à la première et expliquer le maximum de variance restante.

---

# 48. Covariance, valeurs propres et vecteurs propres

On centre les données :

\[
X_c=X-\mu
\]

Puis :

\[
\Sigma
=
\frac{1}{n}X_c^TX_c
\]

On résout :

\[
\Sigma v=\lambda v
\]

Les :

\[
v
\]

sont les directions principales.

Les :

\[
\lambda
\]

représentent la variance expliquée dans ces directions.

---

# 49. Nouvelles features PCA

Supposons :

```text
engine_size
horsepower
weight
```

Une composante pourrait être :

\[
PC_1
=
0.55\,engine\_size
+
0.60\,horsepower
+
0.58\,weight
\]

Elle pourrait représenter une notion globale de :

```text
vehicle size / power
```

Les coefficients sont appelés **loadings**.

---

# 50. Standardisation avant PCA

PCA est très sensible à l'échelle.

Si :

```text
age ≈ 20–80
income ≈ 20 000–200 000
```

le revenu dominera la variance.

On standardise généralement :

\[
x_j'
=
\frac{x_j-\mu_j}{\sigma_j}
\]

avant PCA.

---

# 51. Limite importante de PCA

PCA maximise :

\[
Var(Xw)
\]

et non :

\[
\text{capacité prédictive envers }y
\]

Une composante qui explique beaucoup de variance peut donc être inutile pour prédire la target.

---

# 52. Target Encoding

Pour une variable catégorielle :

```text
Neighborhood
```

on peut remplacer chaque catégorie par une statistique calculée à partir de la target.

Exemple :

| Neighborhood | Price |
|---|---:|
| A | 300 |
| A | 350 |
| A | 400 |
| B | 150 |
| B | 170 |

Alors :

\[
TE(A)=350
\]

\[
TE(B)=160
\]

Conceptuellement :

\[
\boxed{
TE(c)
\approx
E[Y|X=c]
}
\]

---

# 53. Pourquoi Target Encoding est utile

Particulièrement pour les variables à forte cardinalité :

```text
ZIP code
product_id
store_id
movie_id
neighborhood
```

One-Hot Encoding :

\[
10\,000\text{ catégories}
\rightarrow
10\,000\text{ features}
\]

Target Encoding :

\[
10\,000\text{ catégories}
\rightarrow
1\text{ feature}
\]

---

# 54. Risque d'overfitting avec Target Encoding

Supposons une catégorie apparaissant une seule fois :

| category | target |
|---|---:|
| rare_123 | 100 |

Alors :

\[
TE(rare_{123})=100
\]

Le modèle récupère pratiquement directement la target.

C'est extrêmement dangereux.

---

# 55. Smoothing

On combine :

- moyenne de la catégorie ;
- moyenne globale.

Une formule possible :

\[
TE(c)
=
\frac{
n_c\mu_c+m\mu
}{
n_c+m
}
\]

où :

- \(n_c\) = nombre d'exemples de la catégorie ;
- \(\mu_c\) = moyenne de la catégorie ;
- \(\mu\) = moyenne globale ;
- \(m\) = force du smoothing.

Si :

\[
n_c\gg m
\]

alors :

\[
TE(c)\approx\mu_c
\]

Si :

\[
n_c\ll m
\]

alors :

\[
TE(c)\approx\mu
\]

On fait donc moins confiance aux catégories rares.

---

# 56. Out-of-Fold Target Encoding

Pour éviter qu'une ligne utilise sa propre target, on peut construire l'encodage avec des folds.

Exemple :

```text
Fold 1:
calcul des moyennes sur folds 2–5
encodage du fold 1

Fold 2:
calcul des moyennes sur folds 1,3,4,5
encodage du fold 2
```

Ainsi :

\[
\boxed{
\text{une observation n'utilise pas directement son propre }y
}
\]

---

# 57. Gradient Boosting

Après les arbres et Random Forest, on peut utiliser le **Gradient Boosting**.

Différence principale :

## Random Forest

Arbres relativement indépendants :

```text
Tree 1 ─┐
Tree 2 ─┼→ moyenne
Tree 3 ─┤
Tree 4 ─┘
```

## Gradient Boosting

Arbres entraînés séquentiellement :

```text
Tree 1
  ↓
erreurs
  ↓
Tree 2 corrige
  ↓
erreurs restantes
  ↓
Tree 3 corrige
  ↓
...
```

---

# 58. Formulation du Boosting

On part d'un modèle :

\[
F_0(x)
\]

Puis :

\[
F_1(x)
=
F_0(x)
+
\eta h_1(x)
\]

Puis :

\[
F_2(x)
=
F_1(x)
+
\eta h_2(x)
\]

Finalement :

\[
\boxed{
F_M(x)
=
F_0(x)
+
\eta
\sum_{m=1}^{M}h_m(x)
}
\]

où chaque nouveau modèle cherche à corriger les erreurs du précédent.

---

# 59. XGBoost

XGBoost est une implémentation très performante de Gradient Boosted Trees.

Exemple :

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05
)
```

---

# 60. `n_estimators`

Correspond approximativement au nombre d'arbres de boosting.

Trop peu :

\[
\rightarrow underfitting
\]

Trop :

\[
\rightarrow risque\ d'overfitting
\]

---

# 61. `learning_rate`

Dans :

\[
F_m(x)
=
F_{m-1}(x)
+
\eta h_m(x)
\]

\[
\eta
\]

est le learning rate.

Petit learning rate :

- chaque arbre apporte une petite correction ;
- on utilise généralement davantage d'arbres.

Grand learning rate :

- chaque arbre modifie beaucoup la prédiction ;
- entraînement plus agressif.

On observe souvent un compromis :

\[
\boxed{
\text{petit learning rate}
\leftrightarrow
\text{plus d'arbres}
}
\]

---

# 62. Early Stopping

On peut arrêter l'entraînement lorsque la performance de validation ne s'améliore plus.

Exemple :

```text
50 arbres  → MAE 25
100 arbres → MAE 20
150 arbres → MAE 18
200 arbres → MAE 18
250 arbres → MAE 19
```

Continuer n'apporte plus d'amélioration.

On arrête donc autour du meilleur point.

Cette idée se retrouve énormément en Deep Learning.

---

# 63. Vision complète d'un workflow ML

On peut maintenant assembler toutes les notions.

```text
                    RAW DATA
                       │
                       ▼
                TRAIN / VALID SPLIT
                       │
                       ▼
              FEATURE ENGINEERING
            ratios / dates / groups
           interactions / clusters
                       │
                       ▼
                 PREPROCESSING
           ┌───────────┴───────────┐
           │                       │
       NUMERICAL               CATEGORICAL
           │                       │
       Imputation              Imputation
           │                       │
       Scaling si utile        Encoding
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
                  PIPELINE
                       │
                       ▼
                 MODEL CHOICE
            Decision Tree
            Random Forest
            XGBoost
                       │
                       ▼
               CROSS-VALIDATION
                       │
                       ▼
             HYPERPARAMETER TUNING
                       │
                       ▼
                  FINAL MODEL
                       │
                       ▼
                    TEST
                       │
                       ▼
                 PRODUCTION
```

---

# 64. La règle d'or : aucune information du futur

Toutes les notions du cours convergent vers une règle très importante :

\[
\boxed{
\text{une prédiction doit être construite uniquement avec des informations réellement disponibles au moment de cette prédiction}
}
\]

Cela vaut pour :

- les features ;
- l'imputation ;
- le scaling ;
- l'encodage ;
- PCA ;
- K-Means ;
- target encoding ;
- sélection de features ;
- tuning des hyperparamètres.

---

# 65. Le vrai objet que l'on déploie

On pense souvent que le modèle est :

```text
RandomForest.pkl
```

Mais ce n'est pas réellement le cas.

Le système prédictif complet est :

\[
\boxed{
\text{feature engineering}
+
\text{preprocessing}
+
\text{model}
}
\]

Par exemple :

```text
raw request
   ↓
parse features
   ↓
imputation
   ↓
categorical encoding
   ↓
feature transformations
   ↓
model inference
   ↓
post-processing
   ↓
API response
```

C'est particulièrement important en MLOps.

---

# 66. Training Pipeline vs Inference Pipeline

Il faut garantir une cohérence entre :

```text
TRAINING
```

et :

```text
PRODUCTION
```

Si pendant l'entraînement :

\[
x
\xrightarrow{\phi}
z
\xrightarrow{model}
y
\]

alors en production, on doit appliquer exactement :

\[
x_{\text{prod}}
\xrightarrow{\phi}
z_{\text{prod}}
\xrightarrow{model}
\hat y
\]

avec les **mêmes paramètres appris**.

Exemples :

- même moyenne d'imputation ;
- mêmes catégories du OneHotEncoder ;
- même scaler ;
- mêmes axes PCA ;
- mêmes centroïdes K-Means ;
- même logique de feature engineering.

---

# 67. Ce que cela préfigure en MLOps

Les pipelines sklearn donnent une première intuition de problèmes plus larges :

```text
raw data
   ↓
feature transformations
   ↓
model inference
   ↓
prediction
```

Dans un système réel :

```text
API request
   ↓
validation
   ↓
feature preprocessing
   ↓
model server
   ↓
postprocessing
   ↓
response
```

Puis autour :

```text
versioning
monitoring
CI/CD
deployment
observability
drift detection
rollback
```

Le modèle n'est qu'une partie du système.

---

# 68. Lien avec les systèmes LLM

La même logique apparaît avec les applications LLM.

```text
user query
   ↓
preprocessing
   ↓
embedding
   ↓
vector search
   ↓
reranking
   ↓
prompt construction
   ↓
LLM inference
   ↓
output validation
```

Comme avec sklearn :

\[
\boxed{
\text{la qualité dépend de toute la pipeline, pas seulement du modèle final}
}
\]

---

# 69. Feature Engineering vs Deep Learning

Dans le ML classique :

\[
X
\xrightarrow{\text{humain}}
features
\xrightarrow{model}
y
\]

Une grande partie du travail consiste à créer une bonne représentation.

Dans le Deep Learning :

\[
X
\xrightarrow{\text{layers}}
representation_1
\xrightarrow{}
representation_2
\xrightarrow{}
\dots
\xrightarrow{}
y
\]

Le réseau apprend lui-même une partie importante des représentations.

Exemple pour une image :

```text
pixels
 ↓
edges
 ↓
textures
 ↓
shapes
 ↓
objects
```

C'est ce qu'on appelle souvent **representation learning**.

---

# 70. Les distinctions fondamentales à retenir

## `fit` vs `transform`

\[
fit
=
\text{apprendre des paramètres}
\]

\[
transform
=
\text{appliquer les paramètres appris}
\]

---

## Parameters vs Hyperparameters

\[
parameters
=
\text{appris automatiquement}
\]

\[
hyperparameters
=
\text{choisis avant l'entraînement}
\]

---

## Train vs Validation vs Test

\[
train
=
\text{apprendre}
\]

\[
validation
=
\text{choisir / ajuster}
\]

\[
test
=
\text{évaluer finalement}
\]

---

## Preprocessing vs Feature Engineering

\[
preprocessing
=
\text{rendre les données exploitables}
\]

\[
feature\ engineering
=
\text{améliorer leur représentation}
\]

---

## Bagging vs Boosting

Random Forest :

\[
\text{modèles parallèles}
\rightarrow
\text{moyenne}
\]

Boosting :

\[
\text{modèles séquentiels}
\rightarrow
\text{correction progressive}
\]

---

## Supervised vs Unsupervised

Supervised :

\[
(X,y)
\rightarrow
model
\]

Exemples :

- Decision Tree ;
- Random Forest ;
- XGBoost.

Unsupervised :

\[
X
\rightarrow
structure
\]

Exemples :

- K-Means ;
- PCA.

---

# 71. Les concepts essentiels à savoir expliquer sans notes

Après ces trois cours, il faut pouvoir expliquer clairement :

1. **Features et target**
2. **Training vs inference**
3. **Train / validation / test**
4. **Loss et MAE**
5. **Généralisation**
6. **Underfitting**
7. **Overfitting**
8. **Decision Trees**
9. **Random Forest**
10. **Paramètres vs hyperparamètres**
11. **Missing values**
12. **Imputation**
13. **Categorical variables**
14. **One-Hot Encoding**
15. **Ordinal Encoding**
16. **fit / transform / fit_transform**
17. **Pipeline**
18. **ColumnTransformer**
19. **Cross-validation**
20. **Data leakage**
21. **Target leakage**
22. **Feature engineering**
23. **Mutual Information**
24. **Interactions et ratios**
25. **Group transforms**
26. **K-Means**
27. **PCA**
28. **Target Encoding**
29. **Gradient Boosting**
30. **XGBoost**
31. **Learning rate**
32. **Early stopping**

---

# 72. Carte mentale finale

```text
                         MACHINE LEARNING
                               │
                               ▼
                     problème supervisé
                         X  ─────→ y
                               │
                               ▼
                     construire les données
                               │
               ┌───────────────┴────────────────┐
               │                                │
        PREPROCESSING                    FEATURE ENGINEERING
               │                                │
       missing values                   ratios / interactions
       categorical data                 dates / groups
       scaling                          K-Means / PCA
       encoding                         target encoding
               │                                │
               └───────────────┬────────────────┘
                               │
                               ▼
                            PIPELINE
                               │
                               ▼
                       TRAIN / VALIDATION
                               │
                               ▼
                     choix du modèle
             ┌─────────────────┼─────────────────┐
             │                 │                 │
        Decision Tree     Random Forest       XGBoost
             │                 │                 │
             └─────────────────┴─────────────────┘
                               │
                               ▼
                       Cross-Validation
                               │
                               ▼
                     Hyperparameter tuning
                               │
                               ▼
                       Final evaluation
                               │
                               ▼
                            TEST
                               │
                               ▼
                         PRODUCTION
```

---

# 73. Le principe le plus important

Si on devait résumer les trois cours avec une seule idée :

\[
\boxed{
\text{Le Machine Learning n'est pas seulement entraîner un modèle.}
}
\]

Il faut construire un processus dans lequel :

\[
\boxed{
\text{données}
+
\text{représentation}
+
\text{préprocessing}
+
\text{modèle}
+
\text{évaluation correcte}
}
\]

forment un ensemble cohérent.

Le véritable objectif est :

\[
\boxed{
\text{apprendre une fonction qui généralise sur des données futures}
}
\]

et non simplement :

\[
\boxed{
\text{obtenir un score élevé sur les données que l'on possède déjà}
}
\]

C'est cette distinction qui relie directement les bases Kaggle au Machine Learning réel, puis au MLOps et aux systèmes IA en production.
