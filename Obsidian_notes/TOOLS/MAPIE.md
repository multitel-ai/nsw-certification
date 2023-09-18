Topics: [[@DATA QUALITY]]
Sources : [MAPIE - Model Agnostic Prediction Interval Estimator — MAPIE 0.6.5 documentation](https://mapie.readthedocs.io/en/latest/index.html)
Projets : [[Grand Défi 6 - Une IA digne de confiance pour les systèmes critiques]]  
Tags : #framework #library  
Date : 2023-09-12
***
# Model Agnostic Prediction Interval Estimator

## EnbPI (Ensemble batch Prediction Intervals)
For time series (the order of the samples matters)
The coverage guarantee offered by the various resampling methods based on the jackknife strategy, and implemented in MAPIE, are only valid under the “exchangeability hypothesis”. It means that the probability law of data should not change up to reordering.