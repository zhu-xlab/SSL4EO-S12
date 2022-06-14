# SSL4EO-S12
SSL4EO-S12: a large-scale mutilmodal multitemporal dataset for self-supervised learning in Earth observation

### News
- [Jun14, 2022] The dataset has been published at [mediaTUM](https://mediatum.ub.tum.de/1660427).

### Dataset access
The SSL4EO-S12 dataset is openly accessible at [mediaTUM](https://mediatum.ub.tum.de/1660427).

### Pre-trained models

<table>
  <tr>
    <th>arch</th>
    <th>top-1 ImageNet</th>
    <th colspan="2">linear evaluation</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>77.0%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-S/8</td>
    <td>79.7%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/16</td>
    <td>78.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ViT-B/8</td>
    <td>80.1%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_small_12_p16</td>
    <td>77.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p16_pretrain/dino_xcit_small_12_p16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_small_12_p8</td>
    <td>79.2%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_small_12_p8_pretrain/dino_xcit_small_12_p8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p16</td>
    <td>78.8%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p16_pretrain/dino_xcit_medium_24_p16_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>xcit_medium_24_p8</td>
    <td>80.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_xcit_medium_24_p8_pretrain/dino_xcit_medium_24_p8_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
  <tr>
    <td>ResNet-50</td>
    <td>75.3%</td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_linearweights.pth">linear weights</a></td>
    <td><a href="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain_eval_linear_log.txt">logs</a></td>
  </tr>
</table>


TODOs:
- [ ] add pre-trained weights
- [ ] organize the codes


