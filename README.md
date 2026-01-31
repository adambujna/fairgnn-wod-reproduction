# fairGNN-WOD: Fair Graph Learning Without Demographics

This repository is dedicated to a reproduction of the paper ***fairGNN-WOD:
Fair Graph Learning Without Demographics*** by Wang, Liu, Pan, Liu, Saeed, Qiu & Zhang
(2025).
It was created for the **FACT: Fairness, Accountability, Correctness,
and Transparency** course at the **University of Amsterdam
(UvA)**.
The goal is to reproduce the results of the paper,
which presents a solution for achieving fairness in Graph Neural Networks
(GNNs) without the need for complete demographic information.

## Overview

Graph Neural Networks (GNNs) can potentially perform unfairly
(e.g., different prediction accuracy or true positive rate) across different demographic groups.
Because of this, many efforts have been made to ensure more fair results for different groups.
Due to privacy or legal reasons, the procurement of demographic information is not always feasible,
which complicates attempts to achieve fair results.

*fairGNN-WOD: Fair Graph Learning Without Demographics (Wang, et al., 2025)* 
proposes a method of analyzing and mitigating societal bias in GNNs without knowledge of demographic information.
In addition,
the paper also puts forward techniques at expressing the trade-off between prediction accuracy and fairness.
On several benchmark datasets the proposed methods perform comparably in prediction accuracy
while achieving a higher level of fairness than baselines.

## Paper Citation

If you use this repository in your work, please cite the original paper:

Wang, Z., Liu, F., Pan, S., Liu, J., Saeed, F., Qiu, M., & Zhang,
W. fairGNN-WOD: Fair Graph Learning Without Demographics.
In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI-25 (pp. 556–564).
```
@inproceedings{ijcai2025p63,
  title     = {fairGNN-WOD: Fair Graph Learning Without Complete Demographics},
  author    = {Wang, Zichong and Liu, Fang and Pan, Shimei and Liu, Jun and Saeed, Fahad and Qiu, Meikang and Zhang, Wenbin},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {556--564},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/63},
  url       = {https://doi.org/10.24963/ijcai.2025/63},
}
```


---


## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- The authors of *fairGNN-WOD: Fair Graph Learning Without Demographics* for their work on the fairGNN-WOD framework.

- The University of Amsterdam for providing the course and platform for this project.
