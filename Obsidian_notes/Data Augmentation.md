# Synthetic Data Vault (data augmentation)
- Synthetic data generators
	- Single Table datasets
	- Complex, Multi-table, Relational datasets
	- Multi-type, Multi-variate timeseries datasets
- Metrics for synthetic data evaluation

## SDV library
Maj récente (~2semaines), donc les exemples sur le site et ceux du ReadMe ne sont plus totalement valables. Donc à vérifier et modifier pour que ça fonctionne
=> On obtient des data synthétiques générées apd d'un synthetizer Gaussian.. Les données récupérées ne sont pas très représentatives des données initiales

=> continuer en retirant les valeurs qui sont <300 dans les data réelles pour la création de données synthétiques

## DoppleGANger
https://github.com/fjxmlzn/DoppelGANger
https://www.kdnuggets.com/2022/06/generate-synthetic-timeseries-data-opensource-tools.html

Librairie: GRETEL => Timeseries with DGAN



# Developpment
## GANgenerator
#### Tabular Data
Ne fonctionne pas, on obtient des résultats complètements hors de la courbe initiale 
=> Activer le paramètre de post-processing pour filtrer les bottom/top quantile => résultats déjà plus ressemblants aux données initiales

## SDGym
=> permet d'effectuer une analyse (benchmark) des synthetizers de la librairie "Synthetic Data Vault" (SDV) en utilisant des bases de données disponibles ou notre propre base de données.