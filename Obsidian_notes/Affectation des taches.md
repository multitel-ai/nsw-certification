
# DATA

| TASK                      | Bérangère | Florian | Manu |
| ------------------------- | --------- | ------- | ---- |
| [[Sintel Tools  ]]        |           |         | X    |
| TOOLS/[[AI Fairness 360]] | X         |         |      |
| biais/mlp_bias            | X         |         |      |
| MAPIE                     | X         |         |      |
| DA/SDV Single Table (KO)  |           | X       |      |
| DA/SDV Sequantial (KO)    |           | X       |      |
| DA/Augment TS (KO)        |           | X       |      |
| DA/Tsaug (~KO)            |           | X       |      |
| PyOD                      |           |         | X    |
| [[PUNCC  ]]                       |           |         |    X  |







## SOA

## TOOLS
[[Sintel Tools]] : Time Series Toolbox by MIT
[Fetching Title#zt7m](https://pyod.readthedocs.io/en/latest/index.html)  : Outlier/anomaly detection for tabular data
[GitHub - datamllab/tods: TODS: An Automated Time-series Outlier Detection System](https://github.com/datamllab/tods) Outlier/anomaly detection for multivariate time series
[GitHub - donlnz/nonconformist: Python implementation of the conformal prediction framework.](https://github.com/donlnz/nonconformist) - Conformal prediction
[GitHub - deel-ai/puncc: 👋 Puncc is a python library for predictive uncertainty quantification using conformal prediction.](https://github.com/deel-ai/puncc) Conformal Prediction (DEEL Project)



### Fait main 
- file biais/mlp_biais.ipynb : highlights differences of quality of prediction between subgroups (according to chosen feature). Plots the number of samples in each subgroups. Plots the boxplots and dispersion histograms of mae in each subgroups, for ground truth and predicted data.


# Résumé de ce qu'on a

Le but de ce workshop est de chercher différentes méthodes de certification d'IA. 
Application : prédire la puissance consommée en temps réel d'un drone.
Données : tabulaires, récoltées lors de vols de drone :
- batterie (courant, tension)
- vent (vitesse, angle)
- position (x, y, z) + orientation (x, y, z, w)
- vitesse (x, y, z) + accélération (x, y, z)
- timestamps 
- paramètres de vol (route, altitude, vitesse, charge)

## Modèles utilisés
### Histogram gradient boosting regressor
### Multi Layer Perceptron


## Certification des données
### Biais
- On a pas le même nombre de données pour certains groupes. Par exemple on pourrait avoir plus de vols avec une certaine masse.
-> On plot la distribution selon les groupes
Solution (sur les données, peu importe le modèle) :
- oversampling / undersampling 
### Séparation du jeu de données
En training / validation / testing set 
Il faut vérifier que les jeux de données sont similaires, que le jeu de données test n'est pas plus "facile" par exemple.
-> tests statistiques pour vérifier qu'ils proviennent de la même distribution (outil galaad [[wings_data_quality.pdf]])
### Data augmentation 
Sur time series
On peut la certifier aussi : 
- en vérifiant que les données crées suivent la même distribution 

## Certification des modèles
### Biais 
- Le modèle performe mieux sur certaines sous groupes de données
-> on plot l'erreur en fonction des sous groupes 
Comment réguler ces biais ? (solution sur le modèle)
- donner plus de poids aux données des sous groupes défavorisés 
### Incertitude
- Il faut l'évaluer 