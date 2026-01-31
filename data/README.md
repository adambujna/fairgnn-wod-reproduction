# Data

The datasets used are the DBLP dataset `dblp.bin` (Weis, Naumann and Brosy, 2006) 
and the Pokec-z / Pokec-n datasets `pokec_z.bin`, `pokec_n.bin` (Takac and Zabovsky, 2012), (or course-provided forms of them).

To download the datasets we used, run the following script from the `data` directory:
```bash
sh download_datasets.sh
```
and unzip the data
```bash
unzip data.zip
```
>Or unzip `data_pt.zip` for the PyTorch object versions of the graphs.



Alternatively, you can manually download the dataset from:
```
https://huggingface.co/datasets/AdamB2/fact-gnn-wod-data/resolve/main/data.zip
or
https://huggingface.co/datasets/AdamB2/fact-gnn-wod-data/resolve/main/data_pt.zip
```

---
### References
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
