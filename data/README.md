# Data

The datasets used are the DBLP dataset `dblp.pt` (Weis, Naumann and Brosy, 2006) 
and the Pokec-z / Pokec-n datasets `pokec_z.pt`, `pokec_n.pt`, subsets of the Pokec dataset (Takac and Zabovsky, 2012), 
(or course-provided forms of them), and the credit dataset Credit dataset `credit.pt` (Yeh and Lien, 2009).

To download the datasets we used, run the following script from the `data` directory:
```shell
sh download_datasets.sh
```
and unzip the data
```shell
unzip data_pt.zip
```
>Or unzip `data.zip` for the `dgl` object versions of the graphs.



Alternatively, you can manually download the dataset from:
```
https://huggingface.co/datasets/AdamB2/fact-gnn-wod-data/resolve/main/data_pt.zip
or
https://huggingface.co/datasets/AdamB2/fact-gnn-wod-data/resolve/main/data.zip
```

---
### References
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert systems with applications, 36(2), 2473-2480.
```
@article{yeh2009comparisons,
  title={The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients},
  author={Yeh, I-Cheng and Lien, Che-hui},
  journal={Expert systems with applications},
  volume={36},
  number={2},
  pages={2473--2480},
  year={2009},
  publisher={Elsevier}
}
```

Weis, M., Naumann, F., & Brosy, F. (2006, June). A duplicate detection benchmark for XML (and relational) data. In Proc. of Workshop on Information Quality for Information Systems (IQIS).
```
@inproceedings{weis2006duplicate,
  title={A duplicate detection benchmark for XML (and relational) data},
  author={Weis, Melanie and Naumann, Felix and Brosy, Franziska},
  booktitle={Proc. of Workshop on Information Quality for Information Systems (IQIS)},
  year={2006}
}
```

Takac, L., & Zabovsky, M. (2012, May). Data analysis in public social networks. In International scientific conference and international workshop present day trends of innovations (Vol. 1, No. 6).
```
@inproceedings{takac2012data,
  title={Data analysis in public social networks},
  author={Takac, Lubos and Zabovsky, Michal},
  booktitle={International scientific conference and international workshop present day trends of innovations},
  volume={1},
  number={6},
  year={2012}
}
```
