Topics: [[@AI CERTIFICATION]] 
Sources :[GitHub - IBM/ai-360-Toolkit Series](https://github.com/IBM/ai-360-toolkit-explained) 
Projets : 
Tags : #tools #library 
Date : 2023-08-11
***
# Objectives
The capabilities of AI are not hidden. Intelligence of machines is undergoing major transformation by continuous self-learning improvements. However, these AI models are still a black-box and are often questioned for their decisions by the clients. Research is faster than ever on improving and optimisation of the algorithms, but this alone won’t suffice. The conversations around building trust on AI is often a point of interest with the client for developers advocates, Sales and Marketing team which stand at the frontline with them. Hence, it becomes the important aspect to look into. Imagine owning a Computer Vision Company that deals with building AI Classification Models for Healthcare Industry to diagnose cancer using MRIs, CT scans, X-rays, etc which aids doctor in taking decisions. It is difficult for a doctor to rely on the diagnosis suggested by an AI model-a black box when a person’s life is involved. Therefore, building Trusted AI Pipelines has become increasingly important with the sudden shoot of AI Applications.

# Building a [[Trustworthy AI Pipelines]] - Architecture from IBM
![[Pasted image 20230811094048.png|1600]]

# Three-Open source Toolkits
## [AI Fairness 360](https://github.com/Trusted-AI/AIF360)
This extensible open-source toolkit can help you examine, report, and mitigate discrimination and bias in machine learning models throughout the AI application lifecycle. Containing over 70 fairness metrics and 10 state-of-the-art bias mitigation algorithms developed by the research community, it is designed to translate algorithmic research from the lab into the actual practice of domains as wide-ranging as finance, human capital management, healthcare, and education.
**RELEASED** : Sept22
**[Examples](https://github.com/Trusted-AI/AIF360/tree/master/examples)**
## [AI Explainability 360](https://github.com/Trusted-AI/AIX360)
This extensible open source toolkit can help you comprehend how machine learning models predict labels by various means throughout the AI application lifecycle. Containing eight state-of-the-art algorithms for interpretable machine learning as well as metrics for explainability, it is designed to translate algorithmic research from the lab into the actual practice of domains as wide-ranging as finance, human capital management, healthcare, and education.
**RELEASED** : Jul23
**[Examples](https://github.com/Trusted-AI/AIX360/tree/master/examples)**
## [Adversarial Robustness 360 Toolbox](https://developer.ibm.com/open/projects/adversarial-robustness-toolbox/)
The Adversarial Robustness Toolbox is designed to support researchers and developers in creating novel defense techniques, as well as in deploying practical defense of real-world AI systems. Researchers can use the Adversarial Robustness Toolbox to benchmark novel defense against the state-of-the-art. For developers, the library provides interfaces which support the composition of comprehensive defense systems using individual methods as building blocks.
**RELEASED** : Jul23
[Examples](https://github.com/Trusted-AI/adversarial-robustness-toolbox/tree/main/examples)


# Demos from IBM Watson Studio

- [Bias mitigation of AI models using AI fairness 360 toolkit](https://github.com/IBM/bias-mitigation-of-machine-learning-models-using-aif360)
- [Unveiling Machine Fraud Prediction Decision with AI Explainability 360](https://github.com/IBM/unveiling-machine-fraud-prediction-decision-with-ai-explainability-360)
- [# Predict an event with fairness, explainability & robustness using AI fairness 360 toolkit](https://github.com/IBM/predict-an-event-with-fairness-explainability-robustness-using-ai-360-toolkit)