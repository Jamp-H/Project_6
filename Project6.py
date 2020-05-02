###############################################################################
#
# AUTHOR(S): Joshua Holguin, Jacob Christiansen
# DESCRIPTION: program that will implement stochastic gradient descent algorithm
# for a one layer neural network with edits made to the regularization parameter
# VERSION: 0.0.1v
#
###############################################################################
###############################################################################
from sklearn.preprocessing import scale
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import numpy as np

## TODO:
## compare trained models
## create function for baseLine and Dense model
## See DOC

def getNormX(data):
    xValuesUnscaled = data[:,1:]
    x_sc = xValuesUnscaled/255
    return x_sc


def getY(data):
    return data[:,0]

# Function: conventional
# INPUT ARGS:
# X_mat - scaled X matrix for a data set
# y_vec - corresponding outputs for our X_mat
# hidden_layers - Vector of hidden layers
# num_epochs - integer value for the number of epochs wanted
# data_set - string value decribing the data set being ran (Test, Train, Subtrain, ect.)
# Return: array of history objects containing data about the models created
def run_NN(X_mat, y_vec, hidden_value_list , num_epochs, data_set, split):
    # set model variable to keep track on which number model is being ran

    '''
    # set model for single layered NN Dense
    model = keras.Sequential([
    keras.layers.Flatten(input_shape=(np.size(X_mat, 1), )), # input layer
    keras.layers.Dense(hidden_value_list[0], activation='relu'), # hidden layer
    keras.layers.Dense(hidden_value_list[1], activation='relu'), # hidden layer
    keras.layers.Dense(hidden_value_list[2], activation='relu'), # hidden layer
    keras.layers.Dense(hidden_value_list[3], activation='relu'), # hidden layer
    keras.layers.Dense(10, activation='softmax') # output layer
    ])
    '''
    # create model Convilutinal
    model = keras.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu',
                        input_shape = (16,16,1) ),
    keras.layers.Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(hidden_value_list[0], activation='relu'), # hidden layer
    keras.layers.Dense(hidden_value_list[1], activation='relu'), # hidden layer
    keras.layers.Dense(hidden_value_list[2], activation='relu'), # hidden layer
    keras.layers.Dense(hidden_value_list[3], activation='relu'),  # hidden layer
    keras.layers.Dense(10, activation='softmax') # output layer
    ])
    
    
    # compile the model
    model.compile(optimizer='adadelta',
                  loss= "categorical_crossentropy",
                  metrics=['accuracy'])

    # fit the model
    print(f"\nModel {data_set}")
    print("==============================================")
    model_data = model.fit(
                                x=X_mat,
                                y=y_vec,
                                epochs=num_epochs,
                                batch_size=128,
                                validation_split=split)

    return model_data, model


def main():

    # initilize variables
    num_epochs = 10
    hidden_values_dense = [784, 270, 270, 128]
    hidden_values_convolutional = [6272, 9216, 128]

    np.random.seed(2)

    # Get data in matrix format
    zip_train = np.genfromtxt("zip.train", delimiter=" ")

    X_sc = getNormX(zip_train)

    y_vec = getY(zip_train)

    # reshape so each row of matrix is a 16x16 matrix
    X_sc = np.reshape(X_sc[:,:], (X_sc.shape[0], 16, 16, 1))

    fold_ids = np.arange(2)

    fold_vec = np.random.permutation(np.tile(fold_ids,len(y_vec))[:len(y_vec)])
    
    model_fold_data = []
    
    for fold_num in fold_ids:
        x_train = X_sc[fold_num != fold_vec]
        y_train = y_vec[fold_num != fold_vec]

        x_test = X_sc[fold_num == fold_vec]
        y_test = y_vec[fold_num == fold_vec]

        y_train = keras.utils.to_categorical(y_train, num_classes = 10)
        y_test = keras.utils.to_categorical(y_test, num_classes = 10)

        model_data = run_NN(x_train, y_train, hidden_values_dense, num_epochs, "Cross Fold Training", 0.2)
        #convol_model = run_NN(x_train, y_train, hidden_values_convolutional, num_epochs, "Training")

        dense_model_data = model_data[0]
        dense_model = model_data[1]

        history_dense = dense_model_data.history

        best_epochs = np.argmin(history_dense["val_loss"]) + 1
        
        train_set_model = run_NN(x_train, y_train, hidden_values_dense, best_epochs, "Training Set", 0.0)
        
        train_model_data = train_set_model[0]
        train_model = train_set_model[1]
        
        dense_accu = train_model.evaluate(x_test, y_test)
        
        dense_fold_data = {
                  "accu": dense_accu,
                  "Fold": fold_num + 1,
                  "history": history_dense
                }
        
        model_fold_data.append(dense_fold_data)
    
    color_index = 0
    fold_num = 0
    color_array = ['red', 'blue', 'orenge', 'green', 'yellow']
    
    for model in model_fold_data:
        plt.plot(np.arange(best_epochs), model['history']['val_loss'], color=color_array[color_index], label='dense fold {fold_num}')
        min_y = np.amin(model['history']['val_loss'])
        min_x = np.argmin(model['history']['val_loss'])
        plt.plot(min_x, min_y, marker='o', color=color_array[color_index])
        color_index = (color_index + 1) % len(color_array)
        fold_num = fold_num + 1
        
    plt.show()
main()
