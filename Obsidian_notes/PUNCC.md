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


# TEST Fligth 210 = tabular data with 275 samples

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

# TEST Adaptative EnbPI method
## Models: HistGradientBoostingRegressor
Total (Test set)
**MemoryError: Unable to allocate 40.0 GiB for an array with shape (139006, 38614) and data type float64**

![[Pasted image 20230913101857.png]]
Marginal coverage: 0.93 Average width: 105.27

# TEST CV+ method
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

# CV+ method

![[Pasted image 20230913104629.png]]
Marginal coverage: 0.81 Average width: 144.92


# Test sur IID
