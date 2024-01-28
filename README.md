# Progressive Feature Adjustment for Semi-supervised Learning from Pretrained Models

This project contains the implementation of our work for semi-supervised learning from pretrained models:
    
> Progressive Feature Adjustment for Semi-supervised Learning from Pretrained Models,   
> Hai-Ming Xu, Lingqiao Liu, Hao Chen, Ehsan Abbasnejad and Rafael Felix,   
> *Accepted to ICCV workshop 2023*

## Abstract
As an effective way to alleviate the burden of data annotation, semi-supervised learning (SSL) provides an attractive solution due to its ability to leverage both labeled and unlabeled data to build a predictive model. While significant progress has been made recently, SSL algorithms are often evaluated and developed under the assumption that the network is randomly initialized. This is in sharp contrast to most vision recognition systems that are built from finetuning a pretrained network for better performance. While the marriage of SSL and a pretrained model seems to be straightforward, recent literature suggests that naively applying state-of-the-art SSL with a pretrained model fails to unleash the full potential of training data. In this paper, we postulate the underlying reason is that the pretrained feature representation could bring a bias inherited from the source data, and the bias tends to be magnified through the self-training process in a typical SSL algorithm. To overcome this issue, we propose to use pseudo-labels from the unlabelled data to update the feature extractor that is less sensitive to incorrect labels and only allow the classifier to be trained from the labeled data. More specifically, we progressively adjust the feature extractor to ensure its induced feature distribution maintains a good class separability even under strong input perturbation. Through extensive experimental studies, we show that the proposed approach achieves superior performance over existing solutions.

## Installation and Datasets
The running environment and datasets are identical to the [Self-Tuning codebase](https://github.com/thuml/Self-Tuning/tree/master?tab=readme-ov-file#dependencies)

## Training

```
bash runs/run_cars.sh
bash runs/run_aircraft.sh
bash runs/run_cub.sh
```

## Acknowledgement

We thank [Self-Tuning codebase](https://github.com/thuml/Self-Tuning/tree/master) for their impressive work and open-sourced projects.

## Citation
```bibtext
@inproceedings{xu2023progressive,
  title={Progressive Feature Adjustment for Semi-supervised Learning from Pretrained Models},
  author={Xu, Hai-Ming and Liu, Lingqiao and Chen, Hao and Abbasnejad, Ehsan and Felix, Rafael},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop},
  pages={3292--3302},
  year={2023}
}
```
