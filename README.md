# Learning for lensless mask-based Imaging

### [Project Page](https://waller-lab.github.io/LenslessLearning)
This code is based on the paper: "Learned reconstructions for practical
mask-based lensless imaging" available here: (https://www.osapublishing.org/oe/fulltext.cfm?uri=oe-27-20-28075&id=420747)

### Setup:
Clone this project using:
```
git clone https://github.com/Waller-Lab/LenslessLearning.git
```

The dependencies can be installed by using:
```
conda env create -f environment.yml
source activate lensless_learning
```
In addition, the LPIPS package is needed (this is used in the loss function during training). Instructions for installing LPIPS can be found here: (https://github.com/richzhang/PerceptualSimilarity)

### Loading in the models
The pre-trained models can be downloaded [here](https://drive.google.com/a/berkeley.edu/file/d/1aIgWXVt_najoS5ccBglJLEC3WYSFSRX8/view?usp=sharing)

Jupyter Notebook: [pre-trained reconstructions.ipynb](https://github.com/Waller-Lab/LenslessLearning/blob/master/pre-trained%20reconstructions.ipynb)
* Loads in the pre-trained models and runs reconstructions on sample lensless images.
* Initializes un-trained models and shows output images before training.  Changes model parameters to the pre-loaded parameters and shows sample reconstructions 

### Dataset 
* The full Lensless Learning Dataset can be found here: (https://waller-lab.github.io/LenslessLearning/dataset.html)
* In addition, the 'in the wild' images taken without a computer monitor can be found here: (https://drive.google.com/drive/folders/1dtyxApqryiXbpqLSSUKreCVKfjQcT7pS?usp=sharing)

