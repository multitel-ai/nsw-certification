## Link tool :
[GitHub - deel-ai/puncc: 👋 Puncc is a python library for predictive uncertainty quantification using conformal prediction.](https://github.com/deel-ai/puncc) Conformal Prediction (DEEL Project)

# TODO
- [x] Check dataset 'lourd'

# INTRO
Uncertainty Quantification (UQ), building trustworthy and informative prediction intervals. 
To address these industrial needs, we propose to apply Conformal Prediction (CP), a framework that can provide probabilistic guarantees for any underlying predictive model.
# Conformal Prediction
Conformal prediction enables to transform point predictions into interval predictions with high probability of coverage. 
The figure below shows the result of applying the split conformal algorithm on a linear regressor.
![[Pasted image 20230913055434.png]]

## Differentes methodes:
The currently implemented conformal regression procedures are the following:

- `deel.puncc.regression.SplitCP`: Split Conformal Prediction
- `deel.puncc.regression.LocallyAdaptiveCP`: Locally Adaptive Conformal Prediction
- `deel.puncc.regression.CQR`: Conformalized Quantile Regression
- `deel.puncc.regression.CvPlus`: CV + (cross-validation)
- `deel.puncc.regression.EnbPI`: Ensemble Batch Prediction Intervals method
- `deel.puncc.regression.aEnbPI`: locally adaptive Ensemble Batch Prediction Intervals method


Base on paper : [Robust Gas Demand Forecasting With Conformal Prediction](https://proceedings.mlr.press/v179/mendil22a/mendil22a.pdf)

# Definitions
## PI = Prediction interval
## Marginal Coverage: 
regression_mean_coverage
## Average width: 
metrics.regression_sharpness


# TEST 1

## TEST SPLIT CP
### Model : RandomForestRegressor: 
#### Total
Marginal coverage: 0.9 Average width: 148.14
![[Pasted image 20230913104757.png]]
Marginal coverage: 0.96 Average width: 152.61

### Model : HistGradientBoostingRegressor: 

Total (Test set)
Marginal coverage: 0.9 Average width: 153.24

![[Pasted image 20230913094411.png]]
Marginal coverage: 0.96 Average width: 150.81


## TEST CQR : Conformalized Quantile Regression
### Model : 2 HistGradientBoostingRegressors with quantile loss
Total (Test set)
Marginal coverage: 0.9 Average width: 150.62

![[Pasted image 20230913095643.png]]
Marginal coverage: 0.92 Average width: 127.31

## TEST LocallyAdaptiveCP
### Models: 2 HistGradientBoostingRegressors (mu and sigma)
Total (Test set)
Marginal coverage: 0.9 Average width: 145.03
![[Pasted image 20230913094945.png]]
Marginal coverage: 0.91 Average width: 125.28

## TEST EnbPI: Ensemble Batch Prediction Intervals method
### Models: HistGradientBoostingRegressor
Total (Test set)
**MemoryError: Unable to allocate 40.0 GiB for an array with shape (139006, 38614) and data type float64**

![[Pasted image 20230913100728.png]]

Marginal coverage: 0.95 Average width: 134.65

## TEST Adaptative EnbPI method
## Models: HistGradientBoostingRegressor
Total (Test set)
**MemoryError: Unable to allocate 40.0 GiB for an array with shape (139006, 38614) and data type float64**

![[Pasted image 20230913101857.png]]
Marginal coverage: 0.93 Average width: 105.27

## TEST CV+ method
## Models: HistGradientBoostingRegressor
Total (Test set)
Crash Kernel (memory)

4![[Pasted image 20230913103815.png]]
Marginal coverage: 0.97 Average width: 142.12


# Test sur 'LOURD' (1 vol / payload=750)

## SplitCP

![[Pasted image 20230913093910.png]]
Marginal coverage: 0.85 Average width: 150.81

## CQR : Conformalized Quantile Regression


![[Pasted image 20230913095841.png]]

Marginal coverage: 0.83 Average width: 155.66
## LocallyAdaptiveCP
![[Pasted image 20230913095137.png]]

Marginal coverage: 0.82 Average width: 160.37

## TEST EnbPI: Ensemble Batch Prediction Intervals method

![[Pasted image 20230913100949.png]]
Marginal coverage: 0.78 Average width: 134.65

## Adaptative EnbPI method
![[Pasted image 20230913102633.png]]
Marginal coverage: 0.81 Average width: 144.18

## CV+ method

![[Pasted image 20230913104629.png]]
Marginal coverage: 0.81 Average width: 144.92



# Test data nws (erreur de 10%)

## SplitCP
### Test
![[Pasted image 20230913121646.png]]
Marginal coverage: 0.96 Average width: 149.56
### Lourd
![[Pasted image 20230913122249.png]]
Marginal coverage: 0.74 Average width: 149.56
### DA
![[Pasted image 20230913140147.png]]
Marginal coverage: 0.72 Average width: 149.56


## CQR : Conformalized Quantile Regression
## TEST
![[Pasted image 20230913140534.png]]
Marginal coverage: 0.92 Average width: 113.45
## LOURD
![[Pasted image 20230913140928.png]]
Marginal coverage: 0.78 Average width: 146.75
## DA
![[Pasted image 20230913141154.png]]
Marginal coverage: 0.63 Average width: 114.64

## Locally Adaptive Conformal Prediction
### TEST
![[Pasted image 20230913142534.png]]
Marginal coverage: 0.89 Average width: 104.42

### LOURD
![[Pasted image 20230913142623.png]]
Marginal coverage: 0.77 Average width: 151.36
### DA
![[Pasted image 20230913142726.png]]
Marginal coverage: 0.63 Average width: 115.82

## CV+ method
### TEST
![[Pasted image 20230913143204.png]]
Marginal coverage: 0.98 Average width: 143.08
### LOURD
![[Pasted image 20230913143307.png]]
Marginal coverage: 0.77 Average width: 142.37
### DA
![[Pasted image 20230913143406.png]]

Marginal coverage: 0.79 Average width: 150.21

## EnbPI: Ensemble Batch Prediction Intervals method
### TEST
![[Pasted image 20230913144101.png]]
Marginal coverage: 0.97 Average width: 134.06
### LOURD
![[Pasted image 20230913144345.png]]
Marginal coverage: 0.75 Average width: 134.06
### DA
![[Pasted image 20230913144508.png]]
Marginal coverage: 0.7 Average width: 134.06
## Locally adaptative Ensemble Batch Prediction Intervals method
### TEST
![[Pasted image 20230913145202.png]]
Marginal coverage: 0.89 Average width: 92.05
### LOURD
![[Pasted image 20230913145646.png]]
Marginal coverage: 0.77 Average width: 139.7
### DA
![[Pasted image 20230913145921.png]]
Marginal coverage: 0.58 Average width: 102.61
