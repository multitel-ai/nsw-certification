# Nantes Workshop Certification

## Description du projet

### Contexte
Ce projet fait partie d'un summer workshop organisé par TRAIL ([Trusted AI Labs](https://trail.ac/)), un institut qui rassemble  des chercheurs travaillant sur l’IA au sein des universités de la Fédération Wallonie-Bruxelles et des Centres de Recherche Agréés.  L’Institut TRAIL souhaite faire avancer la recherche sur des sujets stratégiques tels que les interactions homme-machine, l’intelligence artificielle explicable (XAI), ou l’IA de confiance protégeant les données privées.

### But
Le but de ce projet est de **CERTIFIER** un système estimant la puissance instantanée utilisée par un drone durant son vol. Ce système a été développé dans le cadre du projet WINGS par [Multitel](https://www.multitel.be/). L'objectif est maintenant de garantir l'utilisation d'une IA digne de confiance. Ceci passe par la **Certification** de l'ensemble du processus permettant d'obtenir et maintenir le composant IA/ML du système proposé. Cette certification couvre un grand nombre d'aspect, allant de la collecte des données au monitoring du système une fois celui-ci mis en production. Parmi les aspects certifiables de l'IA, nous nous sommes concentrés sur la **qualité des données (biais)** ainsi que **l'explicabilité, la robustesse et la gestion de l'incertitude** de nos modèles.

## Code 

#### Modèles IA
Pour répondre à la problématique de prédiction de puissance instantanée consomée par un drone durant son vol, nous avons utilisé deux modèles :
- Histogram gradient boosting regressor (HGBR), de la librairie [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html#sklearn.ensemble.HistGradientBoostingRegressor)
- Un ensemble de Multi Layers Perceptron avec Monte Carlo Dropout, inspiré du [Shift Challenge](https://github.com/Shifts-Project/shifts/tree/main/vpower) sur la prédiction de puissance instantanée d'un paquebot. 

Leur code est disponible dans le dossier *code preparation*.

#### Data quality
Pour évaluer la qualité des données, nous souhaitons savoir si elles contiennent des biais. Le code *mlp biais* met en valeur les différences de distributions au sein d'une variable (comme le poids par exemple), et les différences de performances du modèle selon cette variable (**fairness**). 

Le code *violin plots* montre les différences de distributions de la puissance au sein de différents sous jeux de données. Cela permet de définir un sous jeu de données comme "Out Of Distribution" (OOD) et de voir si le modèle performe aussi bien sur ce dernier. 

#### Data Augmentation

Il est possible de générer des données synthétiques à partir de données réelles afin d'augmenter notre base de données et ainsi de tester notre modèle sur des données nouvelles et parfois différentes.
Pour se faire, différentes librairies ont été testées:

- Synthetic Data Vault ([SDV](https://sdv.dev/SDV/)), permet la génération de données synthétiques en considérant les données soit comme des tableaux (*SDV_SingleTable*), soit comme des time series (*SDV_sequential*).
- [AugmentTS](https://github.com/DrSasanBarak/AugmentTS/tree/main) utilise des Deep Generative Models pour permettre l'augmentation des données mais peut également être utilisée pour la prédiction dans les time series.
- [Tsaug](https://tsaug.readthedocs.io/en/stable/) exploite les données sous forme de time series.

Les codes de ces librairies sont disponibles dans le dossier *DA*.
Le dossier *Models* contient les sauvegardes des modèles entrainés et utilisés pour la data augmentation.

#### MAPIE
Utilisation de la librairie [MAPIE](https://mapie.readthedocs.io/en/latest/index.html) et de son time series regressor pour estimer les intervalles de prédiction avec la méthode EnbPI. 

#### Uncertainty
Pour le modèle de DL (l'ensemble de MLP), 400 inférences sont réalisées pour une entrée (20 modèles initialisés 20 fois). Le modèle sort une moyenne et une déviation standard de gaussienne représentant la distribution de probabilité de la prédiction. Pour évaluer l'incertitude, les courbes de rétention sont calculées.

Pour le modèle ML (le HGBR), l'incertitude est quantifiée à l'aide de méthodes formelles implémentées dans la librairie [PUNCC](https://github.com/deel-ai/puncc). Elle permet de calculer la marginal coverage (le pourcentage de valeurs réelles contenues dans l'intervalle de prediction) et la largeur moyenne d'intervalle. 

## Installation
Le projet contient différent dockers pour éxécuter le code contenu dans des notebooks jupyter. Nous avons utilisé VSCode et son extension Jupyter Notebook pour les ouvrir dans le container directement. \
Les notes Obsidian peuvent être chargées dans l'application [Obsidian](https://obsidian.md/). 

## Usage 
Les données doivent être placées dans un dossier *Data*. 

Exemple de graphe obtenu avec la locally adaptive Ensemble Batch Prediction Intervals method de la librairie Puncc (dans *Uncertainty/ML/test puncc new data*) :
![output](https://github.com/multitel-ai/nsw-certification/assets/144004765/1d6feda5-c9bb-4d2a-a1e8-7571afb8eb41)

## Auteurs
- Emmanuel Jean (jean@multitel.be)
- Bérangère Burgunder (burgunder@multitel.be)
- Florian Facchin 

## Statut du projet
Le projet est terminé.
