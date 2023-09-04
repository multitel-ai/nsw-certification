Topics: [[@UNCERTAINTY QUANTIFICATION]]
Projets : [[Recherche Axe 2 - Une IA digne de confiance]]
Tags : #soa 
Date : 2022-10-21
***

# Concepts

![[Pasted image 20230220095836.png|1500]]

![[Pasted image 20230220100005.png|1500]]


## Methodologies
- [Bayesian-Methods](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#bayesian-methods)
- [Ensemble-Methods](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#ensemble-methods)
- [Sampling/Dropout-based-Methods](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#samplingdropout-based-methods)
- [Auxiliary-Methods/Learning-loss-distributions](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#auxiliary-methodslearning-loss-distributions)
- [Data-augmentation/Generation-based-methods](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#data-augmentationgeneration-based-methods)
- [Dirichlet-networks/Evidential-deep-learning](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#dirichlet-networksevidential-deep-learning)
- [Deterministic-Uncertainty-Methods](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#deterministic-uncertainty-methods)
- [Quantile-Regression/Predicted-Intervals](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#quantile-regressionpredicted-intervals)
- [Calibration](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#calibration)
- [Conformal Predictions](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning#conformal-predictions)


----
MIT video :
- recalibration
![[Pasted image 20230719112242.png]]
- MC Dropout
- ![[Pasted image 20230719112132.png]]
- deep ensembles  ----> BATCH ENSEMBLE (more efficient by sharing parameters)
 ![[Pasted image 20230719112108.png]]
- SWAG + Laplace
- Rank-1 BNN
- MIMO configuration
---




-  [[SOA MC Dropout]]
 - Deterministic Variational Inference (DVI)
 - [[SOA Conformal Prediction]]
 - Calibration
 

# Ressources
OLD (3 ans)
- [GitHub - Literature survey, paper reviews, experimental setups and a collection of implementations for baselines methods for predictive uncertainty estimation in deep learning models.](https://github.com/ahmedmalaa/deep-learning-uncertainty)

NEW (updated)
- [GitHub - ENSTA-U2IS/awesome-uncertainty-deeplearning: This repository contains a collection of surveys, datasets, papers, and codes, for predictive uncertainty estimation in deep learning models.](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning)

MLL
- [GitHub | Awesome-LLM-Robustness: a curated list of Uncertainty, Reliability and Robustness in Large Language Models](https://github.com/jxzhangjhu/Awesome-LLM-Uncertainty-Reliability-Robustness)


# Datasets /Baselines
- [GitHub - MUAD | Multiple Uncertainties for Autonomous Driving, a benchmark for multiple uncertainty types and tasks ](https://github.com/ENSTA-U2IS/MUAD-Dataset)
- [https://github.com/google/uncertainty-baselines/tree/main](https://github.com/google/uncertainty-baselines/tree/main)


# Blogs
- [Uncertainty quantification // van der Schaar Lab](https://www.vanderschaar-lab.com/uncertainty-quantification/)
- [Uncertainty Quantification in Artificial Intelligence-based Systems - KDnuggets](https://www.kdnuggets.com/2022/04/uncertainty-quantification-artificial-intelligencebased-systems.html)
- [Uncertainty Quantification for Neural Networks | |Medium- 2021](https://medium.com/uncertainty-quantification-for-neural-networks/uncertainty-quantification-for-neural-networks-a2c5f3c1836d)

# Trainings / Lectures /Tutorials
- [Introduction to Uncertainty in Deep Learning - Google AI Brain Team](https://www.gatsby.ucl.ac.uk/~balaji/balaji-uncertainty-talk-cifar-dlrl.pdf) 
- [GitHub - Uncertainty deep learning: Baseline](https://github.com/cpark321/uncertainty-deep-learning)
- [Towards Data Science - Get Uncertainty Estimates in Regression Neural Networks for Free](https://towardsdatascience.com/get-uncertainty-estimates-in-neural-networks-for-free-48f2edb82c8f)
- 

# Reviews / Surveys/Benchmark
- [[A Review of Uncertainty Quantification in Deep Learning]]
- [Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to Concepts and Methods](https://arxiv.org/abs/1910.09457)
- [[Arxiv - Jan 22] A Survey of Uncertainty in Deep Neural Networks](https://arxiv.org/abs/2107.03342)
- [[ArXiv Oct 22] MUAD: Multiple Uncertainties for Autonomous Driving, a benchmark for multiple uncertainty types and tasks](https://arxiv.org/abs/2203.01437)
- [[ArXiv Feb23] A Survey on Uncertainty Quantification Methods for Deep Neural Networks: An Uncertainty Source Perspective](https://arxiv.org/abs/2302.13425)
- [[Oct22] Trustworthy clinical AI solutions: a unified review of uncertainty quantification in deep learning models for medical image analysis](https://hal.science/hal-03806630/file/UQ_review_HAL.pdf)


Conformal prediction
- [[ArXiv Dec22] A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)

# Papers
[Thesis 2016- Uncertainty in Deep Learning](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)

[[arXiv 2021] Exploring Uncertainty in Deep Learning for Construction of Prediction Intervals](https://arxiv.org/abs/2104.12953)

[[arXiv] What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://arxiv.org/abs/1703.04977)

[[arXiv Ap22] A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement](https://arxiv.org/abs/2204.09308)

[Notes on the Behavior of MC Dropout](https://arxiv.org/pdf/2008.02627.pdf)

[Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)

[A Simple Approach to Improve Single-Model Deep Uncertainty via Distance-Awareness | Papers With Code](https://paperswithcode.com/paper/a-simple-approach-to-improve-single-model)

[[2107.07511] A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://arxiv.org/abs/2107.07511)

[[arXiv 202207] MAPIE: an open-source library for distribution-free uncertainty quantification](https://arxiv.org/abs/2207.12274)

Post -hoc method
[[arXiv Dec 22] Post-hoc Uncertainty Learning using a Dirichlet Meta-Model](https://arxiv.org/abs/2212.07359)
[GitHub - maohaos2/PosthocUQ](https://github.com/maohaos2/PosthocUQ)

Conformal Prediction and Comformal Risk Control

[Confident Object Detection via Conformal Prediction and Conformal Risk Control](https://hal.science/hal-04063441v1/document)

[[Angelopoulos-2022] Conformal Risk Control](https://arxiv.org/pdf/2208.02814.pdf)

[DEEL Puncc application- Robust Gas Demand Forecasting With Conformal Prediction](https://proceedings.mlr.press/v179/mendil22a/mendil22a.pdf)




# Videos
- [MIT 6.S191: Uncertainty in Deep Learning - YouTube](https://www.youtube.com/watch?v=veYq6EWZyVc)
- [MIT 6.S191: Robust and Trustworthy Deep Learning - YouTube](https://www.youtube.com/watch?v=kIiO4VSrivU&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=6) 
- [Improving Generalization of Monte Carlo Dropout Based DNN Ensemble Model for Speech Enhancement and - YouTube](https://www.youtube.com/watch?v=vaksBs5JBHI)


# Open Libraries
- [GitHub - A library for Bayesian neural network layers and uncertainty estimation in Deep Learning extending the core of PyTorch](https://github.com/IntelLabs/bayesian-torch)

- [GitHub - ENSTA-U2IS| TorchUncertainty: A PyTorch Library for benchmarking and leveraging predictive uncertainty quantification techniques.](https://github.com/ENSTA-U2IS/torch-uncertainty)
- [GitHub | keras-uncertainty: Utilities to perform Uncertainty Quantification on Keras Models](https://github.com/mvaldenegro/keras-uncertainty)



NLP
[GitHub - Model zoo for different kinds of uncertainty quantification methods used in Natural Language Processing, implemented in PyTorch.](https://github.com/kaleidophon/nlp-uncertainty-zoo)

[GitHub - MAPIE: A scikit-learn-compatible module for estimating prediction intervals.](https://github.com/scikit-learn-contrib/MAPIE/)

AWS - Fortuna
[Fortuna — Fortuna's documentation](https://aws-fortuna.readthedocs.io/en/latest/)

Conformal Prediction and Confomal risk Control
- [GitHub - aangelopoulos/conformal-risk: Conformal prediction for controlling monotonic risk functions. Simple accompanying PyTorch code for conformal risk control in computer vision and natural language processing.](https://github.com/aangelopoulos/conformal-risk)
- [GitHub - scikit-learn-contrib/MAPIE: A scikit-learn-compatible module for estimating prediction intervals.](https://github.com/scikit-learn-contrib/MAPIE)
- [GitHub - DEEL - Puncc - a python library for predictive uncertainty quantification using conformal prediction.](https://github.com/deel-ai/puncc)

WRAPPER for Risk aware ML
[[CAPSA - A Library for Risk-Aware and Trustworthy Machine Learning]]



# Commercial Platform
- [SmartUQ - Quantify Every Uncertainty](https://www.smartuq.com/)
- 