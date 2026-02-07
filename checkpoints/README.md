# Model Checkpoints

This folder contains the trained checkpoints of all three used models.
The fairGNN-WOD model is accordingly split into its two stages, 
where `stage1` contains the VGAE and `stage2` contains the second stage.

To download the pre-trained checkpoints, run the following script from the `checkpoints` directory:
```shell
sh download_checkpoints.sh
```
and unzip the data
```shell
unzip checkpoints.zip
```


Alternatively, you can download the checkpoints manually from:
```
https://huggingface.co/AdamB2/fact-gnn-wod-ckpts/resolve/main/checkpoints.zip
```

---

For details about the models included see the README files included in the source directories of each model.
