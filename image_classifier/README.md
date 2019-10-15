# Image Classification Project

Project code for Udacity's Data Scientist Nanodegree program. In this project, I first developed code for an image classifier built with PyTorch, then you I converted it into a command line application.

### Data

The model will be training using 102 different types of flowers, where there ~20 images per flower to train on.  Then you will use your trained classifier to see if you can predict the type for new images of the flowers.


## Pre-requisites
The Code is written in Python 3.6.9. If you don't have Python installed you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade it using pip.

Additional Packages that are required are: 
- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [MatplotLib](https://matplotlib.org/)
- [Pytorch](https://pytorch.org/), 
- PIL
- json


### Project assets:

- `image_classifier_project.ipynb` Jupyter Notebook including main project
- `image_classifier_project.html` HTML export of the Jupyter Notebook above.
- `function_utils.py` Main Script
- `train.py` Command line tool to train a new network on a data set.
- `predict.py` Command line tool to to predict flower name from an image.


## Example Image for Training

Image categories are found in [cat_to_name.json]() and  flower images you can get in this repository itself or can be downloaded in the gziped tar file [flower_data.tar.gz](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz) from Udacity.

Get flower images:
```bash
mkdir -p flowers && cd flowers
curl https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz | tar xz
```

You should now have **test**, **train** and **valid** directories containing classification directories and flower images under the **flowers** directory.

## Viewing the Jupyter Notebook
In order to better view and work on the Jupyter Notebook I encourage you to use [nbviewer](https://nbviewer.jupyter.org/) . You can simply copy and paste the link to this website and you will be able to see it without any problem. Alternatively you can clone the repository using 
```
git clone https://github.com/rowhitswami/Image-Classification-with-PyTorch/
cd Image-Classification-with-PyTorch/
```
Open terminal in current folder and type:
```
jupyter notebook
```
locate the notebook and run it.


## Command Line Application
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set direcotry to save checkpoints: ```python train.py data_dir --save_dir save_directory```
    * Choose arcitecture (densenet161 or vgg16 available): ```pytnon train.py data_dir --arch "vgg16"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_units 512 256 --epochs 20 ```
    * Use GPU for training: ```python train.py data_dir --gpu```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` with ```/path/to/checkpoint``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```