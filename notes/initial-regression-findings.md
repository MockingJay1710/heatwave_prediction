- en regardant l erreur global MAE, notre modele ne donne pas d ameliorations significatives par rapport a la baseline.
- apres tracage des courbes reel contre predit sur chaque mois, on a remarque que notre modele arrive a suivre en general l'allure du reel mais avec des problemes:
	- le modele est tjrs decale ou en retard, il pose bcp de poids sur les temperature du jour j; ca se voit sur les coefficients accordes aux variables, Tmax et T_mean
	- il arrive a suivre la monotonie mais il est aussi en retard.
	- il n arrive pas a predire les changements brusques de temperatures (donnees insuffisantes ???) + les courbes de predictions sont lisses
	- les pics et les creux sont attenues

- Hypothèse A
La relation est non linéaire.
→ tester un modèle non linéaire plus tard.


Hypothèse B
Les variables locales du jour t
ne contiennent pas assez d'information
sur ce qui arrive demain.
→ ajouter des informations spatiales.


Hypothèse C
On exploite mal la dynamique temporelle.
→ ajouter t-1, t-2, t-3, tendances,
   variations de pression, etc.


Hypothèse D
Certaines variables pourraient être plus
informatives sous forme de variation.
Ex :
pressure(t) - pressure(t-1)
Tmax(t) - Tmax(t-1)
humidity(t) - humidity(t-1)