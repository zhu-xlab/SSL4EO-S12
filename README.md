# SSL4EO-S12
The SSL4EO-S12 dataset is a large-scale multimodal multitemporal dataset for unsupervised/self-supervised pre-training in Earth observation. The dataset consists of unlabeled patch triplets (Sentinel-1 dual-pol SAR, Sentinel-2 top-of-atmosphere multispectral, Sentinel-2 surface reflectance multispectral) from 251079 locations across the globe, each patch covering 2640mx2640m and including four seasonal time stamps.

![ssl4eo-s12](assets/hello.png)

### Access the dataset
- [x] **Raw dataset**: The full SSL4EO-S12 dataset (1.5TB, 500GB for each modality) is accessible at [mediaTUM](https://mediatum.ub.tum.de/1660427). There are some void IDs (gaps in folder names), see `data/void_ids.csv`. Center coordinates of all locations are available [here](https://drive.google.com/file/d/1RyJnGznSbMparS88BhHkXxETf0K-qYqI/view?usp=sharing).
- [x] **Example subset**: An example 100-patch subset (600MB) is available at [Google Drive](https://drive.google.com/file/d/1sRWcYbaWs-efXza6kw03GlJQdZHq5iRN/view?usp=sharing).
- [x] **Compressed dataset**: A compressed 8-bit version (20-50GB for each modality, including an RGB version) is available at [mediaTUM](https://mediatum.ub.tum.de/1702379). The raw 16/32-bit values are normalized by mean and std and converted to uint8. *Note: in our experiments, 8-bit input performs comparably well as 16-bit.*
- [ ] A 50k (random) RGB subset (18GB) is available here (link broken, we are working on it). Sample IDs see `data/50k_ids_random.csv`.

### Collect your own data
Check [`src/download_data`](src/download_data) for instructions to download sentinel or other products from Google Earth Engine.


### Pre-trained models
The pre-trained models with different SSL methods are provided as follows (13 bands of S2-L1C, 100 epochs, input clip to [0,1]).


| SSL method |   Arch   | BigEarthNet | EuroSAT | So2Sat-LCZ42 |                                                   Download                                                  |          |      | Usage |
|:----------:|:--------:|:-----------:|:-------:|:------------:|:-----------------------------------------------------------------------------------------------------------:|:--------:|:----:|:----:|
|    [MoCo](https://github.com/facebookresearch/moco)    | ResNet50 |    91.8%    |  99.1%  |     60.9%    |    [full ckpt](https://drive.google.com/file/d/1OrtPfG2wkO05bimstQ_T9Dza8z3zp8i-/view?usp=sharing)    | [backbone](https://drive.google.com/file/d/1MAe3dCW4hPasSaBMZAVkJVX80LONkrLY/view?usp=sharing) | [logs](https://drive.google.com/file/d/1G66pdvJmeD6Rc-OZdOKA1h2Vnvq_0nnt/view?usp=sharing) | [define model](https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/linear_BE_moco.py#L228-L236), [load weights](https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/src/benchmark/transfer_classification/linear_BE_moco.py#L248-L276) |
|    [MoCo](https://github.com/facebookresearch/moco-v3)        | ViT-S/16 |    89.9%    |  98.6%  |     61.6%    |   [full ckpt](https://drive.google.com/file/d/1Tx07L6OilkfcgE2HWiSXHRmRepCPdn6V/view?usp=sharing)   | [backbone](https://drive.google.com/file/d/1LREGuI6w7Gq6Xm0jFQdxxtp8QkmLvJWk/view?usp=sharing) | [logs](https://drive.google.com/file/d/1f05B85T4Y2-RntfAw42uICKm9mwilHXF/view?usp=sharing) | [define model](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/linear_BE_moco_v3.py#L182-L184), [load weights](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/linear_BE_moco_v3.py#L199-L220) |
|    [DINO](https://github.com/facebookresearch/dino)    | ResNet50 |    90.7%    |  99.1%  |     63.6%    |    [full ckpt](https://drive.google.com/file/d/1iSHHp_cudPjZlshqWXVZj5TK74P32a2q/view?usp=sharing)    | [backbone](https://drive.google.com/file/d/1B4o_NvY7O6fJrvsOUR-7QzLYNpRL1ieA/view?usp=sharing) | [logs](https://drive.google.com/file/d/1VxjT-3n1ckbvnlsF81jZwmm9Wvb3YX0H/view?usp=sharing) | [define model](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/linear_BE_dino.py#L57-L61), [load weights](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/models/dino/utils.py#L92-L103) |
|   [DINO](https://github.com/facebookresearch/dino)         | ViT-S/16 |    90.5%    |  99.0%  |     62.2%    |   [full ckpt](https://drive.google.com/file/d/1CseO5vvMReGlAulm5o4ZgbjUgj8VlAH7/view?usp=sharing)   | [backbone](https://drive.google.com/file/d/1kjQWfPRI5z43EmRkw5fzgHU01hB7E_4H/view?usp=sharing) | [logs](https://drive.google.com/file/d/1eeKrKFMa6akGyXugBRF6-rJ7oTIeZAno/view?usp=sharing) | [define model](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/linear_BE_dino.py#L53-L55), [load weights](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/models/dino/utils.py#L92-L103) |
|     [MAE](https://github.com/facebookresearch/mae)    | ViT-S/16 |    88.9%    |  98.7%  |     63.9%    |    [full ckpt](https://drive.google.com/file/d/1QTBKl1asxgQCNd6bO2azXZNPfoQ3Sazv/view?usp=sharing)   | [backbone](https://drive.google.com/file/d/1hdie-7orFnj5Q1E1C2BudqwQCvMk3Fza/view?usp=sharing) | [logs](https://drive.google.com/file/d/1uJojq9q_fKMdD6cO1YXCPguZYEmfj35s/view?usp=sharing) | [define model](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/linear_BE_mae.py#L232-L236), [load weights](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/linear_BE_mae.py#L238-L259) |
|  [Data2vec](https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec)  | ViT-S/16 |    90.3%    |  99.1%  |     64.8%    | [full ckpt](https://drive.google.com/file/d/1VbIGBwzZYndv4v1vx9FiD6IP-YwsHEns/view?usp=sharing) | [backbone](https://drive.google.com/file/d/1YecuYPAxl1NIzLmsmdbUROjCb5g0t80l/view?usp=sharing) | logs | [define model](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/linear_BE_data2vec.py#L372-L390), [load weights](https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/src/benchmark/transfer_classification/linear_BE_data2vec.py#L406-L553) |

Other pre-trained models:

| SSL method |   Arch   | Input |                                                           Download                                                           |          |      |
|:----------:|:--------:|:----------------:|:----------------------------------------------------------------------------------------------------------------------------:|:--------:|:----:|
|    MoCo    | ResNet18 | S2-L1C 13 bands      |             [full ckpt](https://drive.google.com/file/d/1iWLm7ljQ6tKZiVp47pJUPDe3Un0BUd9o/view?usp=sharing)            | backbone | logs |
|            | ResNet18 | S2-L1C RGB            | [full ckpt](https://drive.google.com/file/d/1HfgXS5VpQA39k8mFrWMbHvYwuT_j6Mbi/view?usp=sharing), [full ckpt ep200](https://drive.google.com/file/d/1U_m39Owahk15Vg1uL1MYbPAmAyUWBKfI/view?usp=sharing) | backbone | logs |
|            | ResNet50 | S2-L1C RGB            |             [full ckpt](https://drive.google.com/file/d/1UEpA9sOcA47W0cmwQhkSeXfQxrL-EcJB/view?usp=sharing)            | backbone | logs |
|            | ResNet50 | S1 SAR 2 bands            |             [full ckpt](https://drive.google.com/file/d/1gjTTWikf1qORJyFifWD1ksk9HzezqQ0b/view?usp=sharing)            | [backbone](https://drive.google.com/file/d/1E5MvVI1SnQneQXe37QAWx_B6aoTiSN24/view?usp=sharing) | logs |

### License
This repository is released under the Apache 2.0 license. The dataset and pretrained model weights are released under the CC-BY-4.0 license.


### Citation
```BibTeX
@article{wang2022ssl4eo,
  title={SSL4EO-S12: A Large-Scale Multi-Modal, Multi-Temporal Dataset for Self-Supervised Learning in Earth Observation},
  author={Wang, Yi and Braham, Nassim Ait Ali and Xiong, Zhitong and Liu, Chenying and Albrecht, Conrad M and Zhu, Xiao Xiang},
  journal={arXiv preprint arXiv:2211.07044},
  year={2022}
}
```
