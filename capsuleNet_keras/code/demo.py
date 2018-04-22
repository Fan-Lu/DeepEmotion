# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:45:18 2018
@author: yuxi
"""

"""Train a simple CNN-Capsule Network on the fer2013 images dataset.
Without Data Augmentation:
"""


from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,Input,Lambda,Reshape
import tensorflow as tf
import numpy as np
import keras
from keras import callbacks
import matplotlib.pyplot as plt
import time
from scipy.misc import imsave
import itertools
from sklearn.metrics import classification_report, confusion_matrix


# the squashing function.
# we use 0.5 in stead of 1 in hinton's paper.
# if 1, the norm of vector will be zoomed out.
# if 0.5, the norm will be zoomed in while original norm is less than 0.5
# and be zoomed out while original norm is greater than 0.5.
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# define our own softmax function instead of K.softmax
# because K.softmax can not specify axis.
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# define the margin loss like hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


class Capsule(Layer):
    """A Capsule Implement with Pure Keras
    There are two vesions of Capsule.
    One is like dense layer (for the fixed-shape input),
    and the other is like timedistributed dense (for various length input).
    The input shape of Capsule must be (batch_size,
                                        input_num_capsule,
                                        input_dim_capsule
                                       )
    and the output shape is (batch_size,
                             num_capsule,
                             dim_capsule
                            )
    Capsule Implement is from https://github.com/bojone/Capsule/
    Capsule Paper: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(1, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(
                name='capsule_kernel',
                shape=(input_num_capsule, input_dim_capsule,
                       self.num_capsule * self.dim_capsule),
                initializer='glorot_uniform',
                trainable=True)

    def call(self, inputs):
        """Following the routing algorithm from Hinton's paper,
        but replace b = b + <u,v> with b = <u,v>.
        This change can improve the feature representation of Capsule.
        However, you can replace
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        with
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        to realize a standard routing.
        """

        if self.share_weights:
            hat_inputs = K.conv1d(inputs, self.kernel)
        else:
            hat_inputs = K.local_conv1d(inputs, self.kernel, [1], [1])

        batch_size = K.shape(inputs)[0]
        input_num_capsule = K.shape(inputs)[1]
        hat_inputs = K.reshape(hat_inputs,
                               (batch_size, input_num_capsule,
                                self.num_capsule, self.dim_capsule))
        hat_inputs = K.permute_dimensions(hat_inputs, (0, 2, 1, 3))

        b = K.zeros_like(hat_inputs[:, :, :, 0])
        for i in range(self.routings):
            c = softmax(b, 1)
            
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            o = self.activation(K.batch_dot(c, hat_inputs, [2, 2]))
            
            if i < self.routings - 1:
                b = K.batch_dot(o, hat_inputs, [2, 3])
                if K.backend() == 'theano':
                    o = K.sum(o, axis=1)

        return o

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

#%%
batch_size = 128
num_classes = 7
epochs = 30
class_names = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

#%%load data
def loadfer2013():
    
    tmp = np.loadtxt("fer2013.csv", dtype=np.str, delimiter=",")
    reader = tmp[1:,1]
    TRAIN_END_POINT = len(reader)
    train_data_x = np.zeros([TRAIN_END_POINT, 48, 48], dtype="uint8")
    for k, data in enumerate(reader):
       pixels_formated = [int(a) for a in data.split(" ")] 
       pixels_in_picture_format = np.reshape(pixels_formated, [48, 48])
       train_data_x[k, :, :] = pixels_in_picture_format
    label = tmp[1:,0].astype(np.uint8)
    return train_data_x, label

def create_plots(history):

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy of CNN+CapsuleNet')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('margin_loss of CNN+CapsuleNet')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('loss.png')
    plt.clf()
    
def plot_confusion_matrix(confusionmatrix, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        confusionmatrix = confusionmatrix.astype('float') / confusionmatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confusionmatrix)

    plt.imshow(confusionmatrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confusionmatrix.max() / 2.
    for i, j in itertools.product(range(confusionmatrix.shape[0]), range(confusionmatrix.shape[1])):
        plt.text(j, i, format(confusionmatrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if confusionmatrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')    

#%%CNN+CapsuleNet
# A common Conv2D model
input_image = Input(shape=(48, 48, 1))
conv1 = Conv2D(64, (3, 3), activation='relu')(input_image)
conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
averagePooling= AveragePooling2D((2, 2))(conv2)
conv3 = Conv2D(128, (3, 3), activation='relu')(averagePooling) 
conv4 = Conv2D(128, (3, 3), activation='relu')(conv3)




"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.
the output of final model is the lengths of 7 Capsule, whose dim=16.
the length of Capsule is the proba,
so the problem becomes a 7 two-classification problem.
"""

x = Reshape((-1, 128))(conv4)
capsule = Capsule(7, 16, 3, True)(x)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(inputs=input_image, outputs=output)
model.summary()
#%%
model.load_weights('trained_model_capsuleNet.h5') 

#%%video

import cv2
CASC_PATH = 'haarcascade_frontalface_default.xml'
SIZE_FACE = 48
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']



cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
def brighten(data,b):
     datab = data * b
     return datab

def format_image(image):
    if len(image.shape) > 2 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imdecode(image, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    faces = cascade_classifier.detectMultiScale(
        image,
        scaleFactor = 1.3,
        minNeighbors = 5
    )

    # None is we don't found an image

    if not len(faces) > 0:
        return None
    max_area_face = faces[0]
    for face in faces:
        if face[2] * face[3] > max_area_face[2] * max_area_face[3]:
            max_area_face = face

    # Chop image to face

    face = max_area_face
    image = image[face[1]:(face[1] + face[2]), face[0]:(face[0] + face[3])]
    # Resize image to network size

    try:
        image = cv2.resize(image, (SIZE_FACE, SIZE_FACE), interpolation = cv2.INTER_CUBIC) / 255.
    except Exception:
        print("[+] Problem during resize")
        return None

    return image



video_capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX



feelings_faces = []
for index, emotion in enumerate(EMOTIONS):
    feelings_faces.append(cv2.imread(emotion + '.png', -1))



while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    a = format_image(frame)
    # Predict result with network
    if(a is None):
        result = None
    else:   
        a = a.reshape(-1, 48, 48, 1)
        result = model.predict(a,verbose=2)
        result = result.reshape(7)

    if result is not None:
        for index, emotion in enumerate(EMOTIONS):
            cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1);
            cv2.rectangle(frame, (130, index * 20 + 10), (130 + int(result[index]*100), (index + 1) * 20 + 4), (255, 0, 0), -1)
        face_image = feelings_faces[np.argmax(result)]
        # Ugly transparent fix
        for c in range(0, 3):
            frame[200:320, 10:130, c] = face_image[:, :, c] * (face_image[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - face_image[:, :, 3] / 255.0)
    # Display the resulting frame

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
