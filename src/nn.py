# -*- coding: utf-8 -*-

"""
Code to train, evaluate and save the Neural Network trained on the cleaned datset. 
Use the cleaned dataset (from CV19_data_cleaner.py). You can set the hyperparameters
for learning_rate_initial and list of layers for the number of Dense layers you want to train

Citations:
Pandas:  McKinney, Proceedings of the 9th Python in Science Conference, Volume 445, 2010.
sklearn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
SMOTEENN: Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning, 
            Guillaume et al., JMLR 18 (17), pp. 1-5, 2017.
Tensorflow: Martín et al., TensorFlow: Large-scale machine learning on heterogeneous systems,
                2015. Software available from tensorflow.org

"""
import sys
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
import tensorflow as tf
import pandas as pd
from imblearn.combine import SMOTEENN

#Hyperparamters
learning_rate_initial = 0.01
#Can add more lists of neuron lengths and sizes as necessary
list_of_layers = [[100,70,50,20]]
print("Training Neural Network")
def train_and_save_nn(learning_rate_initial, list_of_layers):
    path = "..\\build\\cleandata.csv"
    data = pd.read_csv(path)
    y = data.pop('outcome')
    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, y, test_size=0.15, random_state = 0) #Changed from 0.33
    x_train, y_train =  SMOTEENN().fit_resample(x_train, y_train)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=0.1765) #Changed from 0.33
    dataset = tf.data.Dataset.from_tensor_slices((x_train.values, y_train.values))
    train_dataset = dataset.shuffle(len(x_train)).batch(64)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val.values, y_val.values))
    val_dataset = val_dataset.shuffle(len(x_val)).batch(64)
    
    #Defining the plot boolean
    input_arguments = str(sys.argv)
    plot_bool = input_arguments[1]
    if plot_bool == "True":
        plot_bool = True
    else:
        plot_bool = False
    
    def find_callbacks():
        return [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0,
        mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)]
    
    def model_fn(layers):
        layers_list = []
        for i in range(len(layers)):
            # layers_list.append(tf.keras.layers.Dropout(0.2))
            layers_list.append(tf.keras.layers.Dense(layers[i],activation='relu'))
        layers_list.append(tf.keras.layers.Dense(1, activation="sigmoid"))
        model = tf.keras.Sequential(layers_list)
        model.compile(optimizer= tf.keras.optimizers.Adam(),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy', 'Precision','Recall',tf.keras.metrics.AUC(name = "auc")])
        return model
    def run_plot(layers):
        model = model_fn(layers)
        history = model.fit(train_dataset, epochs=10000, verbose = 0, 
                            validation_data = val_dataset, callbacks = find_callbacks())
        model.save("nn_class")
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc']) 
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train_acc', 'val_acc', "train_auc","val_auc"], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss']) 
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        results = model.evaluate(x_test, y_test, verbose = 0)
        y_pred = model.predict(x_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)  
        plt.scatter(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()  
        return results, fpr,tpr, history.history['loss'], history.history['val_loss']
    values_holder = []
    fpr_holder = []
    tpr_holder = []
    loss_holder = []
    val_loss_holder = []
    
    for i in range(len(list_of_layers)):
        values, fpr, tpr, loss, val_loss = run_plot(list_of_layers[i])
        values_holder.append(values)
        fpr_holder.append(fpr)
        tpr_holder.append(tpr)
        loss_holder.append(loss)
        val_loss_holder.append(val_loss)
    
    color_scheme = ['lightcoral', "orange", "palegoldenrod", "seagreen", "deepskyblue", "mediumpurple"]
    
    def loss_plot(loss, val_loss):
        for i in range(len(loss)):
            plt.plot(loss[i], label = 'loss'+str(i), color = color_scheme[i])
            plt.plot(val_loss[i], label = "val_loss"+str(i), ls = '-.', color = color_scheme[i])
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(loc='upper left')
        plt.show()
        return
    
    def roc_plot(fpr, tpr):
        for i in range(len(fpr)):
            plt.scatter(fpr[i], tpr[i], label = str(i), color = color_scheme[i], s = 5)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc='lower left')
        plt.show()
        return    
    if plot_bool:
        loss_plot(loss_holder, val_loss_holder)
        roc_plot(fpr_holder, tpr_holder)
        
    with open("holder.txt", 'a') as f:
        f.write('\n')
        f.write(str(values_holder))
    return

train_and_save_nn(learning_rate_initial, list_of_layers)

    




