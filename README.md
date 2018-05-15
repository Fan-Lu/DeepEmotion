# Live Emotion recognition with CapsNet

This repository is the our project about live emotion recognition using capsule network for the course Big Dat Ecosystem at UF.

## Dependencies

- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [TFLearn](https://github.com/tflearn/tflearn#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## Dataset

We use the [FER-2013 Faces Database](http://www.socsci.ru.nl:8180/RaFD2/RaFD?p=main), a set of 28,709 pictures of people displaying 7 emotional expressions (angry, disgusted, fearful, happy, sad, surprised and neutral).

You have to request for access to the dataset or you can get it on [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

## Usage

```bash
$ python video.py poc
```
## Steps to Run Experiments
Step 1: install the dependencies
Tensorflow: edit the following in your terminate
          conda create -n tensorflow pip python=3.5 
          active tensorflow
          pip install --ignore-installed --upgrade tensorflow
Keras: edit the following comment in your terminate
          pip install keras
OpenCV: pip install opencv-python
Pytorch: conda install pytorch torchvision -c pytorch

Step 2: download the Fer2013 dataset in the following web:
https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Step 3: run the code files in the folders
First, you should download the Emojis and Haarcascade_files in your local folders.
Second, you should download the model which we have already trained, which are h5 files. There are two models we trained one is CNN model and the other is CNN-CapsNet model.
Finally, you can run the demo.py in the folders, and there are other three files, which represent different models’ code. You can also run the code in the src folders, which contain the Pytorch version.
