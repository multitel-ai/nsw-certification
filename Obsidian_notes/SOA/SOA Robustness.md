# Concepts

- **ROBUSTNESS**
	- STABILITY  (// Weak perturbation)
	- AGAINST ADVERSARIAL ATTACK (// Weak perturbation)

--> Robustness evaluation  by WhiteBox Attack
 - Robust training = defense against attack (ex: adversarial training)

![[Pasted image 20230727163130.png|1500]]

ROBUSTNESS TRAINING
- Model Soups
- [[SOA Data Augmentation]]

ROBUSTNESS CERTIFICATION
- [[SOA Formal Methods]]
- [Sci-Hub 2021| Robustness certification with generative models ](https://sci-hub.hkvisa.net/10.1145/3453483.3454100)

**Robustness verification & **Robust training** **
___
- **[[Robustness verification]]** aim to evaluate DNN robustness by providing a theoretically certified **lower bound** of robustness under certain perturbation constraints
- **[[Robust training]]** aim to train DNNs to improve such lower bound.
- Issues (// trade-off):
	- Scalability  - complete verification is NP-Complete
	- Tightness   - relaxation -> 'larger'  lower bound
___


# Ressources

# Trainings / Lectures /Tutorials


# Reviews / Surveys

# Papers

# Videos

# Libraries
[GitHub - Model Soups ](https://github.com/mlfoundations/model-soups) 
	- Averaging weights of multiple fine-tuned models improves accuracy without increasing inference time
 [GitHub - VeriGauge: A united toolbox for running major robustness verification approaches for DNNs](https://github.com/AI-secureVeriGauge)
	- `cnn_cert/`: from [https://github.com/IBM/CNN-Cert](https://github.com/IBM/CNN-Cert)
	- `convex_adversarial/`: from [https://github.com/locuslab/convex_adversarial](https://github.com/locuslab/convex_adversarial)
	- `crown_ibp/`: from [https://github.com/huanzhang12/CROWN-IBP](https://github.com/huanzhang12/CROWN-IBP)
	- `eran/`: from [https://github.com/eth-sri/eran](https://github.com/eth-sri/eran)
	- `recurjac/`: from [https://github.com/huanzhang12/RecurJac-and-CROWN](https://github.com/huanzhang12/RecurJac-and-CROWN)


