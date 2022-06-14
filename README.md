# SSL4EO-S12
SSL4EO-S12: a large-scale mutilmodal multitemporal dataset for self-supervised learning in Earth observation

### News
- [Jun14, 2022] The dataset has been published at [mediaTUM](https://mediatum.ub.tum.de/1660427).

### Dataset access
The SSL4EO-S12 dataset is openly accessible at [mediaTUM](https://mediatum.ub.tum.de/1660427).

### Pre-trained models

| SSL method |   Arch   | BigEarthNet | EuroSAT | So2Sat-LCZ42 |                                                   Download                                                  |          |      |
|:----------:|:--------:|:-----------:|:-------:|:------------:|:-----------------------------------------------------------------------------------------------------------:|:--------:|:----:|
|    MoCo    | ResNet50 |    91.8%    |  99.1%  |     60.9%    |    [full ckpt](https://syncandshare.lrz.de/getlink/fiUTyFN9kvFVhBfFBry6K8wK/B13_rn50_moco_0099_ckpt.pth)    | [backbone](https://syncandshare.lrz.de/getlink/fiYDLhcuBnssotKvLCwUTjad/B13_rn50_moco_0099.pth) | logs |
|            | ViT-S/16 |    89.9%    |  98.6%  |     61.6%    |   [full ckpt](https://syncandshare.lrz.de/getlink/fiMJnvN2F2bi7enxbDduN9Tq/B13_vits16_moco_0099_ckpt.pth)   | [backbone](https://syncandshare.lrz.de/getlink/fi9nXfDQCgM37sBShifgZvDG/B13_vits16_moco_0099.pth) | logs |
|    DINO    | ResNet50 |    90.7%    |  99.1%  |     63.6%    |    [full ckpt](https://syncandshare.lrz.de/getlink/fiEqiTz7JM2TFFxBa8D91mfo/B13_rn50_dino_0095_ckpt.pth)    | [backbone](https://syncandshare.lrz.de/getlink/fiUYZu2N7oNsfHHu6skVHJ3b/B13_rn50_dino_0099.pth) | logs |
|            | ViT-S/16 |    90.5%    |  99.0%  |     62.2%    |   [full ckpt](https://syncandshare.lrz.de/getlink/fi9mUJArfqKWtYeadixqxGfE/B13_vits16_dino_0099_ckpt.pth)   | [backbone](https://syncandshare.lrz.de/getlink/fiXrWtDAdYgEwbWGUBhcjxxc/B13_vits16_dino_0099.pth) | logs |
|     MAE    | ViT-S/16 |    88.9%    |  98.7%  |     63.9%    |    [full ckpt](https://syncandshare.lrz.de/getlink/fiXAvqk1spqizGLKaFpfENSX/B13_vits16_mae_0099_ckpt.pth)   | [backbone](https://syncandshare.lrz.de/getlink/fiKHsA3LyYLV8cUSMWphpUNE/B13_vits16_mae_0099.pth) | logs |
|  Data2vec  | ViT-S/16 |    90.3%    |  99.1%  |     64.8%    | [full ckpt](https://syncandshare.lrz.de/getlink/fiV5t9MAya9UiV3U729ovyPN/B13_vits16_data2vec_0099_ckpt.pth) | [backbone](https://syncandshare.lrz.de/getlink/fi8GpGpL3zXmZ6ETPDTmeLGT/B13_vits16_data2vec_0099.pth) | logs |


TODOs:
- [ ] add pre-trained weights
- [ ] organize the codes


