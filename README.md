# Automatic Infectious Disease Classification Analysis with Concept Discovery
This repository contains code to replicate experiments from the paper:

Automatic Infectious Disease Classification Analysis with Concept Discovery \
by Elena Sizikova, Joshua Vendrow, Xu Cao, Rachel Grotheer, Jamie Haddock, Lara Kassab, Alona Kryshchenko, Thomas Merkh, R. W. M. A. Madushani, Kenny Moise, Annie Ulichney, Huy V. Vo, Chuntian Wang, Megan Coffee, Kathryn Leonard, Deanna Needell  

*Note*: code is based on the following implementation of TB Classifier by (Duong el al. 2021: "Detection of tuberculosis from chest X-ray images: Boosting the performance with vision transformer and transfer learning"):
https://github.com/linhduongtuan/Tuberculosis_ChestXray_Classifier

Steps to run code: 

1. Download tbx11k_vgg16.pth (pretrained model), TBX11K_classification_splits_sub.zip (subset of TBX for fast test) 
and TBX11K_classification_splits.zip (full TBX11K dataset in sorter format) 
\
from https://drive.google.com/drive/folders/1BaCmSiD2-ZhzfvoE12Gnd4JJfRm8ph3j?usp=sharing

2. Install python libraries. We are using: \
torch 1.8.1+cu102 \
torchvision 0.9.1+cu102 \
numpy 1.22.3 \
cv2 4.5.5 

3. (Optional) Train VGG16 classifier by running the train_xray_classifier.ipynb notebook.

4. Run nmf and ssnmf optimization on TBX11K by running the nmf_ssnmf_tbx11k_subset.ipynb notebook.


# BibTeX
```
@article{sizikova2022automatic,
  title={Automatic Infectious Disease Classification Analysis with Concept Discovery},
  author={Sizikova, Elena and Vendrow, Joshua and Cao, Xu and Grotheer, Rachel and Haddock, Jamie and Kassab, Lara and Kryshchenko, Alona and Merkh, Thomas and Madushani, R. W. M. A. and Moise, Kenny and and Ulichney, Annie and Vo, Huy V. and Wang, Chuntian and Coffee, Megan and Leonard, Kathryn and Needell, Deanna},
  journal={arXiv preprint},
  year={2022}
}

```

