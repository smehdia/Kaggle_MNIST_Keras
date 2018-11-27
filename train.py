import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, LeakyReLU, BatchNormalization, Activation, Dropout
from keras import regularizers
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import TrainValTensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.utils import plot_model
import os
from tqdm import tqdm
import shutil

'''
define constant parameters and hyper parameters here
'''
# general parameters
IMG_SIZE = 28
NUM_CHANNELS = 1
# network parameters
NUM_BASE_FILTERS = 16
REGULARIZATION_PARAM = 1e-3
DROPOUT = 0.25
ALPHA_LEAKY_RELU = 0.2
NUM_DENSE = 64
# training options
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
MAX_EPOCHS = 500
FLAG_SAVE_WRONG_VALIDATION_PREDICTIONS = True
# augmentation options
SHEAR_RANGE = 0.1
ZOOM_RANGE = 0.1
ROTATION_RANGE = 20
SHIFT_RANGE = 0.1
# remove logs from previous runs
shutil.rmtree('logs')

class Model_Class:
    def __init__(self, rg, alpha_leaky_relu, num_base_filters, dropout_factor, num_dense_1):
        # model hyper parameters
        self.rg = rg # regularization
        self.alpha_leaky_relu = alpha_leaky_relu  # slope of leaky relu
        self.num_base_filters = num_base_filters # number of first layer filters
        self.dropout_factor = dropout_factor
        self.num_dense_1 = num_dense_1  # number of neurons in dense layer before final dense

    # cnn model definition in keras
    def build_model(self):
        input_image = Input(shape=(28, 28, 1))
        x = Conv2D(filters=self.num_base_filters, kernel_size=(5, 5), kernel_regularizer=regularizers.l2(self.rg),
                   name='first_layer')(input_image)
        x = BatchNormalization()(x)
        x = LeakyReLU(self.alpha_leaky_relu)(x)
        x = MaxPooling2D((2, 2))(x)

        x = Conv2D(filters=2 * self.num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(self.rg))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(self.alpha_leaky_relu)(x)

        x = Conv2D(filters=4 * self.num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(self.rg))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(self.alpha_leaky_relu)(x)

        x = Conv2D(filters=8 * self.num_base_filters, kernel_size=(3, 3), kernel_regularizer=regularizers.l2(self.rg))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(self.alpha_leaky_relu)(x)

        x = Flatten()(x)

        x = Dense(self.num_dense_1)(x)
        x = LeakyReLU(self.alpha_leaky_relu)(x)

        x = Dropout(self.dropout_factor)(x)

        x = Dense(10)(x)
        x = Activation('softmax')(x)

        # build API model in keras
        model = Model(inputs=[input_image], outputs=[x])
        # save model architecture
        plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True)
        # print model summary on the terminal
        model.summary()
        return model

    def data_generator(self):
        # define image generator
        datagen = ImageDataGenerator(
            rotation_range=ROTATION_RANGE,
            width_shift_range=SHIFT_RANGE,
            height_shift_range=SHIFT_RANGE,
            shear_range=SHEAR_RANGE, zoom_range=ZOOM_RANGE)

        return datagen

# create custom callback in order to visualize first layer filters
class draw_first_layer_filters(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        # draw first layer filters
        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])
        first_layer_weights = np.array(layer_dict['first_layer'].get_weights()[0])
        sqrt_num_filters = int(np.ceil(np.sqrt(first_layer_weights.shape[3])))
        for i in range(sqrt_num_filters):
            for j in range(sqrt_num_filters):
                plt.subplot(sqrt_num_filters, sqrt_num_filters, j + sqrt_num_filters * i + 1)
                try:
                    plt.imshow(first_layer_weights[:, :, 0, j + sqrt_num_filters * i], cmap='gray',
                               interpolation='none')
                    plt.axis('off')
                except:
                    continue
        if epoch == 0:
            plt.savefig('first_layer_filters_before_training.png')
        else:
            plt.savefig('first_layer_filters.png')

        plt.close()

if __name__ == "__main__":

    # load model
    try:
        # load train-validation and test data
        X, X_test, y, y_test = np.load('data/X.npy'),np.load('data/X_test.npy'),np.load('data/y.npy'),np.load('data/y_test.npy')
    except:
        raise Exception('Dataset is incomplete (!!! NOTE THAT FIRST YOU SHOULD RUN prepare_dataset.py) !!!')


    cnn = Model_Class(REGULARIZATION_PARAM, ALPHA_LEAKY_RELU, NUM_BASE_FILTERS, DROPOUT, NUM_DENSE)
    cnn_model = cnn.build_model()
    # compile model using Adam optimizer
    optimizer = Adam(LEARNING_RATE)
    cnn_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    # split train-validation data to train and validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    # convert targets to one hot vectors and scale images between [-1, 1]
    y_train_one_hot = to_categorical(y_train,num_classes=10)
    y_val_one_hot = to_categorical(y_val,num_classes=10)
    y_test_one_hot = to_categorical(y_test, num_classes=10)
    X_train = X_train.astype('float32').reshape([-1, 28, 28, 1])
    X_val = X_val.astype('float32').reshape([-1, 28, 28, 1])
    X_test = X_test.astype('float32').reshape([-1, 28, 28, 1])
    X_train = (X_train - 127.5) / 127.5
    X_val = (X_val - 127.5) / 127.5
    X_test = (X_test - 127.5) / 127.5
    # get generator handle
    data_generator = cnn.data_generator()
    data_generator.fit(X_train)
    # define callbacks for saving model, early stopping, visualizing first layer filters
    checkpoint = ModelCheckpoint('./models/model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0, patience=20, verbose=0, mode='auto')
    draw_first_layer_filters_callback = draw_first_layer_filters()
    # training model
    cnn_model.fit_generator(data_generator.flow(X_train, y_train_one_hot, batch_size=BATCH_SIZE), validation_data=(X_val, y_val_one_hot),epochs=MAX_EPOCHS, verbose=1, steps_per_epoch=X_train.shape[0]//BATCH_SIZE,callbacks=[early_stopping,checkpoint,draw_first_layer_filters_callback,TrainValTensorBoard(write_graph=False)])

    if FLAG_SAVE_WRONG_VALIDATION_PREDICTIONS:
        # save wrong predictions of validation data
        # get prediction on validation data
        pred = cnn_model.predict(X_val)
        # convert one hot predictions to integer predictions
        pred = np.argmax(pred, axis=1)
        # get wrong prediction indices
        indices = [i for i, v in enumerate(pred) if pred[i] != y_val[i]]
        subset_of_wrongly_predicted = [X_val[i] for i in indices]
        # prepare path for saving wrong predictions
        if not os.path.isdir("wrong_predictions"):
            os.makedirs('wrong_predictions')
        for i in tqdm(range(len(subset_of_wrongly_predicted))):
            img = subset_of_wrongly_predicted[i].reshape([28, 28])
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            # save in the format number of image _ model prediction _ ground truth
            plt.savefig('wrong_predictions/{}_{}_{}.png'.format(i, pred[indices[i]], y_train[indices[i]]))
            plt.close()

    # show accuracy on test set
    print(cnn_model.evaluate(x=X_test, y=y_test_one_hot))


