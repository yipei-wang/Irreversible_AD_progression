# Implementation of "Learning the irreversible progression trajectory of Alzheimer's disease"

This repository contains the implementation of the paper ["Learning the irreversible progression trajectory of Alzheimer's disease"](https://arxiv.org/pdf/2403.06087), which is accepted at ISBI 2024.

**Overview**
Alzheimer's Disease is an irreversible progress. The risk of an individual subject is monotonic with time. This work applies a regularization scheme to impose the monotonic progression in this longitudinal progression, thus improves the trustworthiness of the resulting model.


**Method**
The implementation of the proposed regularization term is detailed in `train.py`. 
`train_net_wrt_samples` performs standard training procedure regarding samples directly, while `train_net_wrt_subjects` performs training in the subject level to stay consistent with the regularization. 
The regularized training procedure is detailed in `train_net_with_regularization` and `regularize`.


For tabular data, simple multi-layer perceptrons (MLPs) are utilized. Details are in `models.py`.


**Dataset**
We downloaded longitudinal structural MRI, amyloid PET and other clinical data from the [Alzheimer’s Disease Neuroimaging Initiative (ADNI) database](https://adni.loni.usc.edu/). The ADNI is a longitudinal study that was launched in 2003 to track the progression of AD by using clinical and cognitive tests, MRI, FDG-PET, amyloid PET, CSF, and blood biomarkers. More details can be found in the [manuscript](https://arxiv.org/pdf/2403.06087).
