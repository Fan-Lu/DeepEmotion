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
    
#%%
data, label = loadfer2013()
x_train = data[0:28700]
y_train = label[0:28700]
x_validation = data[28709:32298]
y_validation = label[28709:32298]
x_test = data[32298:]
y_test = label[32298:]

#%%
x_train = x_train.reshape(-1, 48, 48, 1).astype('float32') / 255.
x_validation = x_validation.reshape(-1, 48, 48, 1).astype('float32') / 255.
x_test = x_test.reshape(-1, 48, 48, 1).astype('float32') / 255.

y_train = keras.utils.to_categorical(y_train, num_classes)
y_validation = keras.utils.to_categorical(y_validation, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#%%
# A common Conv2D model
input_image = Input(shape=(None, None, 1))
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

# we use a margin loss
log = callbacks.CSVLogger('log.csv')
checkpoint = callbacks.ModelCheckpoint('CapsuleNet_weights-{epoch:02d}.h5',save_best_only=True, save_weights_only=True, verbose=1)
#model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy']) 

history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1, validation_data=(x_validation, y_validation),callbacks=[log,checkpoint])
create_plots(history) # plot accuracy and loss

score = model.evaluate(x_test, y_test, verbose=0)
model.save_weights('trained_model_capsuleNet.h5')
print('Trained model saved to \'trained_model.h5\'')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# we can compare the performance with or without data augmentation
#data_augmentation = False
data_augmentation = False
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by dataset std
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in 0 to 180 degrees
        width_shift_range=0.1,  # randomly shift images horizontally
        height_shift_range=0.1,  # randomly shift images vertically
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        workers=4)
#%%plot confusion matrix
Y_pred = model.predict(x_test,verbose=2)
y_pred = np.argmax(Y_pred,axis=1)
for ix in range(num_classes):
        print (ix, confusion_matrix(np.argmax(y_test,axis=1), y_pred)[ix].sum())
print (confusion_matrix(np.argmax(y_test,axis=1), y_pred))
    
plot_confusion_matrix(confusion_matrix(np.argmax(y_test,axis=1), y_pred), classes=class_names)

#%%
model.load_weights('trained_model_capsuleNet.h5')  
#%%
input_img = model.input

# get the symbolic outputs of each "key" layer (we gave them unique names).
#layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


img_width = 48
img_height = 48
kept_filters = []
for filter_index in range(0, 128):
    # we only scan through the first 200 filters,
    # but there are actually 512 of this block5_cov1 layer
    print('Processing filter %d' % filter_index)
    start_time = time.time()

    # we build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    # 定义一个损失函数，这个损失函数将用于最大化某个指定滤波器的激活值。以该函数为优化目标优化后，我们可以真正看一下使得这个滤波器激活的究竟是些什么东西
    layer_output = conv4 #the first convolusion layer's filter
    if K.image_data_format() == 'channels_first':
        loss = K.mean(layer_output[:, filter_index, :, :])
    else:
        loss = K.mean(layer_output[:, :, :, filter_index])

    # we compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    # 计算出来的梯度进行了正规化，使得梯度不会过小或过大。这种正规化能够使梯度上升的过程平滑进行
    grads = normalize(grads)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    # step size for gradient ascent
    step = 1.

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        input_img_data = np.random.random((1, 1, img_width, img_height))
    else:
        input_img_data = np.random.random((1, img_width, img_height, 1))
    input_img_data = (input_img_data - 0.5) * 20 + 128

    # we run gradient ascent for 20 steps
    # 根据刚刚定义的函数，现在可以对某个滤波器的激活值进行梯度上升，这里是梯度下降的逆向应用，即将当前图像像素点朝着梯度的方向去"增强"，让图像的像素点反过来和梯度方向去拟合
    for i in range(30):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

        print('Current loss value:', loss_value)
        if loss_value <= 0.:
            # some filters get stuck to 0, we can skip them
            break

    # decode the resulting input image
    if loss_value > 0:
        img = deprocess_image(input_img_data[0])
        kept_filters.append((img, loss_value))
    end_time = time.time()
    print('Filter %d processed in %ds' % (filter_index, end_time - start_time))
#%%
# we will stich the best 64 filters on a 8 x 8 grid.
n = 2

# the filters that have the highest loss are assumed to be better-looking.
# we will only keep the top 64 filters.
kept_filters.sort(key=lambda x: x[1], reverse=True)
kept_filters = kept_filters[:n * n]

# build a black picture with enough space for
# our 8 x 8 filters of size 128 x 128, with a 5px margin in between
margin = 2
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin
stitched_filters = np.zeros((width, height, 1))

# fill the picture with our saved filters
for i in range(n):
    for j in range(n):
        img, loss = kept_filters[i * n + j]
        stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img



fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.imshow(stitched_filters[:,:,0],cmap='gray')
# save the result to disk
imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters[:,:,0])


