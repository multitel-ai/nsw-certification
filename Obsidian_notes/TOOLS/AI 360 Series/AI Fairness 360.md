Topics: [[@AI CERTIFICATION]] 
Sources : [AI Fairness 360 video](https://www.youtube.com/watch?v=X1NsrcaRQTE)
Projets : 
Tags : #tools #library 
Date : 2023-09-04
***
# Overview
## Pipeline
![[Pasted image 20230904154752.png]]

- dataset metrics : bias metrics
- pre-processing algorithms : pour réduire les biais trouvés 
- in processing : modifie le code du modèle avec de la régularisation par exemple
- post processing : sur la predicition, pour les rendre plus fair 
- classifier metrics 

## Fairness
Plusieurs définitions donc il faut des outils personnalisés 

## Bias
data biases à cause de :
- prejudice in labels
- undersampling or oversampling
Par exemple le genre, l'ethnicité, la classe social ect 
On pourrait just les retirer mais ils sont très corrélés à d'autres données dont on a besoin en général

# Use 
- Surtout pour detecter et corriger les biais : 
	- from aif360.sklearn.detectors import bias_scan [tutorial\_bias\_advertising.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/tutorial_bias_advertising.ipynb) 
-> marche plutot pour la classification, pas très utile pour la regression 

- Model explaination :
	- [demo\_lime.ipynb](https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_lime.ipynb) LIME Local Interpretable Model-Agnostic Explanations 


# Other tools
![[Pasted image 20230904161406.png]]
![[Pasted image 20230904161439.png]]
