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


## Used/Tried tools for DA
Synthetic Data Vault (SDV) -> Single Table / Sequential (Gaussian & GAN / PAR)
VAE ~ Auto-Encoder (deep_tabular_augmentation)
GAN (tabgan)



GitHub pour la modification des données (jitter, sliding, reversing, ...) : https://github.com/uchidalab/time_series_augmentation/blob/master/example.ipynb
Mind Map des possibles méthodesde DA : https://github.com/AgaMiko/data-augmentation-**review
A voir/faire/coder : 
=> AugmentTS :: Time Series Data Augmentation using Deep Generative Models
=> LSTM-autoencoder with attentions for multivariate time series

### Augment TS ---
en utilisant le "augmenter", on peut soit recréer des points de l'espace latent ou bien des points qui ressemblent à des données qu'on lui donne. Nous, on veut surtout recréer des points qui sont ressemblants à ceux utilisés dans nos sets de données

### Tsaug ---
génération/augmentation des données de vols, on obtient des centaines de vols supplémentaires mais en regardant les valeurs/évolutions de la power_smoothed des vols générés, elles ne semblent pas réalistes du tout.





---------------------
## SDV library
Librairie récente (maj ~2 semaines), donc les exemples sur le site/ GitHub/ ReadMe/... ne sont plus tout à fait valables et doivent être revus/modifiés pour fonctionner.
#### Single Table
-> On obtient des data synthétiques générées apd d'un synthesizer Gaussian => les données récupérées ne sont pas très représentatives des données réelles/ initiales.
-> On obtient des data synthétiques  générées apd d'un synthesizer GAN => les données récupérées sont plus représentatives des données réelles/ initiales. En y ajoutant encore un filtre supplémentaire (savgol_filter), il est possible d'obtenir des graphes pour les différentes valeurs plus resseblantes à celles initialement utilisées.
	-> juste check les valeurs de window et d'order du filtre en fonction de la longueur des TS, pour obtenir de bonnes courbes plutôt réalistes

=>**Plot** de la puissance de l'ensemble des vols (training sur Gaussian et GAN)
![[SDV_SingleTable_Real.png]]
![[SDV_SingleTable_Fake_Gaussian.png]]
![[SDV_SingleTable_Fake_GAN.png]]
=> **Plot** 1 vol après filtrage (ensemble des données générées via GAN et sur les différents features)
![[SDV_SingleTable_Flight_Filtered.png]]
![[SDV_SingleTable_Flight_Filtered_all.png]]

#### Sequential
-> Pareil, obtient des données synthétiques mais qui ne matchent pas avec les données réelles qu'on a utilisées

=> **Plot** de la puissance de l'ensemble des vols (réel et avec un PAR)
![[SDV_Sequential_Real.png]]
![[SDV_Sequential_Fake_PAR.png]]
## AugmentTS (Time Series Data Augmentation using Deep Generative Models)
=> https://github.com/DrSasanBarak/AugmentTS/tree/main
Basic features : 
- Time Series Data Augmentation (using Deep Generative Models)
- Visualizing the Latent Space of Generative Models
- Time Series Forecasting (using Deep Neural Networks)
-> On obtient des données synthétiques (soit avec un nombre random de samples, soit en se basant sur la dimension d'une time series données en entrée). Mais les données (d'un point de vue de la puissance_smoothed), ne sont pas trés représentatives et varient énormément.
Mais les résulats obtenus ressemblent à des TS (>< SDV) et peuvent être plus utiles que les nuages de points qui paraissent aléatoires.


## Tsaug
=> https://github.com/arundo/tsaug/tree/master
Python package for Time Series Augmentation.
-> On obtient des données synthétiques, qui ont été générées (ultra rapide comparé aux autres .. qq 0.1 secondes), mais les données ne semblent pas toutes trés réalistes. Ceci dit, vu les 975 TS qui ont été générées apd des 195, il doit bien y  avoir des TS qui sont réalistes et peuvent être exploitées, utilisées.
=> A check comment il est possible d'analyser ces données et de ne garder que celles qui semblent les plus réalistes/proches des TS initiales du dataset réel.

=> **Plot** de 10 vols réels et 10 vols générés par l'augmenter
![[Tsaug_real_plot.png]]

![[Tsaug_augment_plot.png]]
