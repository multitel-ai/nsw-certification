
# Description du projet
Le but de ce projet est de **CERTIFIER** une un système estimant la puissance instantanée utilisée par un drone durant son vol. Ce système a été développée dans le cadre du projet WINGS par Multitel. L'objectif est maintenant de garantir l'utilisation d'une IA digne de confiance. Ceci passe par la **Certification** de l'ensemble du processus permettant d'obtenir et maintenir le composant IA/ML du système proposé. Cette certification couvre un grand nombre d'aspect, allant de la collecte des données au monitoring du système une fois celui-ci mis en production. Dans le cadre du workshop, nous commencerons donc d'abord par définir les aspects que l'on veux certifier (**data, explicabilité, robustesse, gestion de l'incertitude,..** ). 

Le but de ce projet sera de mettre en avant les méthodes et outils  nécessaires pour certifier le système proposé et les implémenter/modifier afin de constituer un framework de certification pour une tache de regression sur des données tabulaires.


# Ressources
### Communication : Slack
### Code : gitlab(github)-multitel ( à creer)
- dossier prepa (nettoyage code Berangère)
- 1 dossier  par thématique ( Data quality / Model Quality)
	- Sous dossier par travaux
### Documentation 
- Google drive
	- obsidian


# Idées
### 
### Data Quality Assessment  (Data validation & verification)- 
- Data Splits (Training/Validation/Test) independance

- Data correctness
	- Shift/Drift // Out of distribution
	- Error Detection / Anomaly
	- Bias Detection 
	- Annotation quality assessment
- Data completeness and representativeness

#### Verification de la distribution des Datasets + Shift/Drift
- KS Test : [The Kolmogorov-Smirnov Test: A Powerful Tool in AI for Data Analysis | by Siti Khotijah | Jul, 2023 | Medium](https://medium.com/@khotijahs1/the-kolmogorov-smirnov-test-a-powerful-tool-in-ai-for-data-analysis-e3187a317c44#:~:text=The%20Kolmogorov%2DSmirnov%20test%20is,of%20the%20datasets%20being%20compared.)
- [[arXiv Oct22] Explanation Shift: Detecting distribution shifts on tabular data via the explanation space](https://arxiv.org/abs/2210.12369)

- Investigation méthode classique de DA pour regression tabulaire -->> test de distribution ( // qualité de la DA)


#### Shift Detection /
- [GitHub - DFKI-NLP/xai-shift-detection](https://github.com/DFKI-NLP/xai-shift-detection)
- [GitHub - SeldonIO/alibi-detect: Algorithms for outlier, adversarial and drift detection](https://github.com/SeldonIO/alibi-detect)
- [GitHub - steverab/failing-loudly: Code repository for our paper "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift": https://arxiv.org/abs/1810.11953](https://github.com/steverab/failing-loudly)

####  Anomaly Detection

####  Bias Detection



#### Data completeness and representativeness
[[SOA Data Qualiity]]

### Quantification d'incertitude
[MAPIE ](https://mapie.readthedocs.io/en/latest/)
[[arXav Nov 22] Sample-based Uncertainty Quantification with a Single Deterministic Neural Network](https://arxiv.org/abs/2209.08418)
[https://arxiv.org/pdf/2003.02037.pdf](https://arxiv.org/pdf/2003.02037.pdf)

##### ressources
- [[2305.16583] Detecting Errors in Numerical Data via any Regression Model](https://arxiv.org/abs/2305.16583)


#### Prediction Quality Assessment
-  [[SOA-Uncertainty Quantification]]
-  [[SOA - Domain generalization]]  (quantifiable generalization garantees)
- Robustness and stability
- adversarial detection

# Preparation

## DATA Quality
- Input    :    Tabular data ( 25 columns)

- Output :    Energy consomption ( by delta t)
	- Issue   :    Sampling rate instable


## MODEL Quality
- robustness/generalisation :   train et dev_in (M=0,250,500) / dev_out (M = 750)
	- Formal method : [[MLEAP - Delivrable 1 - R&S#Tools overview]]  

## Modéle pour l'estimation de la puissance instantanée

### HistoBoostRegressor
### MC-Dropout MLP 
- [[GD6 - Shift Challenge - Track 2]]





# Participants
- **Emmanuel Jean**|[jean@multitel.be](mailto:jean@multitel.be)
- Bérangere Burgunder  [burgunder@multitel.be](mailto:berangere.burgunder@multitel.be)
 Florian Facchin  [facchin@multitel.be](mailto:berangere.burgunder@multitel.be)

# RoadMap

# Follow-Up

# Réalisation

# Livrables 

# Extentions