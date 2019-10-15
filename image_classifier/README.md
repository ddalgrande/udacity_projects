# Image Classification Project

Project code for Udacity's Data Scientist Nanodegree program. In this project, I first developed the code for an image classifier built with PyTorch, then I converted it into a command line application.


## Data

The model was trained using 102 different types of flowers, where there are ~20 images per flower to train on.

Image categories can be found in [cat_to_name.json]() and  flower images can be downloaded here [flower_data.tar.gz](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz).

Download flower images:
```bash
mkdir -p flowers && cd flowers
curl https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz | tar xz
```


## Pre-requisites

The Code is written in Python 3.6.9. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version you can upgrade it using pip.

Additional Packages that are required are:

- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [MatplotLib](https://matplotlib.org/)
- [Pytorch](https://pytorch.org/)
- PIL
- json


### Project assets:

- `image_classifier_project.ipynb` Jupyter Notebook including the main analysis
- `function_utils.py` script including functions used in train.py and predict.py
- `train.py` command line tool to train a new neural network using transfer learning
- `predict.py` command line tool to predict the name of a flower - the script will also print probabilities 


## Command Line Application

### Train
Train a new network on a data set with `train.py`

Basic usage: ```python train.py data_directory```
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
- Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory```
- Choose architecture: ```python train.py data_dir --arch "vgg13"```
- Set hyperparameters: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20```
- Use GPU for training: ```python train.py data_dir --gpu```


### Predict

Predict flower name from an image with `predict.py` along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: ```python predict.py /path/to/image checkpoint```

Options:
- Return top KK most likely classes: ```python predict.py input checkpoint --top_k 3```
- Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
- Use GPU for inference: ```python predict.py input checkpoint --gpur
