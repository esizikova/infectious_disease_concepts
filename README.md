Code Release for ML4H Submission:: 
# Automatic Infectious Disease Classification Analysis with Concept Discovery


1. Download tbx11k_vgg16.pth (pretrained model), TBX11K_classification_splits_sub.zip (subset of TBX for fast test) 
and TBX11K_classification_splits.zip (full TBX11K dataset in sorter format) 

from https://drive.google.com/drive/folders/1BaCmSiD2-ZhzfvoE12Gnd4JJfRm8ph3j?usp=sharing

2. Install python libraries. We are using:
torch 1.8.1+cu102
torchvision 0.9.1+cu102
numpy 1.22.3
cv2 4.5.5

3. (Optional) Train VGG16 classifier by running the train_xray_classifier.ipynb notebook

4. Run nmf and ssnmf optimization on TBX11K by running the nmf_ssnmf_tbx11k_subset.ipynb notebook





