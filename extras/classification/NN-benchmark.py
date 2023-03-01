<<<<<<< HEAD
import argparse
import pathlib

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
=======
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from qibocal.data import Data 
>>>>>>> b7d4240d55d33ddfc45271aa4679bafff85737ac
import seaborn as sns
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, Normalization
from keras import optimizers
from keras.layers import Layer
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import time
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix 
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
<<<<<<< HEAD
=======
import pathlib
>>>>>>> b7d4240d55d33ddfc45271aa4679bafff85737ac

def results(model,x_train,y_train,x_test,y_test,ml_results,model_name):
    start = time.time()
    model.fit(x_train, y_train)
    stop = time.time()
    training_time = stop-start
    score = model.score(x_test, y_test)
    print("Accuracy", score)
    start = time.time()
    y_pred = model.predict(x_test)
    stop = time.time()
    test_time = (stop - start)/len(x_test)

    ml_results["model"].append(model_name)
    ml_results['accuracy'].append(score)
    ml_results['testing time'].append(test_time)
    ml_results['training time'].append(training_time)
    
    return confusion_matrix(y_test,y_pred,normalize="true")

class RBFLayer(Layer):
        def __init__(self, units, gamma, **kwargs):
            super(RBFLayer, self).__init__(**kwargs)
            self.units = units
            self.gamma = K.cast_to_floatx(gamma)

        def build(self, input_shape):
            self.mu = self.add_weight(name='mu',
                                    shape=(int(input_shape[1]), self.units),
                                    initializer='uniform',
                                    trainable=True)
            super(RBFLayer, self).build(input_shape)

        def call(self, inputs):
            diff = K.expand_dims(inputs) - self.mu
            l2 = K.sum(K.pow(diff, 2), axis=1)
            res = K.exp(-1 * self.gamma * l2)
            return res

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.units)


# Random search 
def model_builder(hp):
        hp_units_1 = hp.Int('units_1', min_value=16, max_value=1056, step=16)
        hp_units_2 = hp.Int('units_2', min_value=16, max_value=1056, step=16)
        activation=hp.Choice("activation", ["relu", "sigmoid", "tanh","RBF"])
        learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2)
        optimizer_choice=hp.Choice("optimizer", ["Adam", "Adagrad","SGD","RMSprop"])
        norm = hp.Boolean('add_normalisation')
        losses = hp.Choice('losses', ['binary_crossentropy','categorical_crossentropy'])
        
        
        model = Sequential()
        if norm:
                model.add(Normalization())
        if activation == "RBF":
                model.add(RBFLayer(hp_units_1,
                        hp.Float("mu", min_value=1e-4, max_value=1),
                        input_shape=(2,)))
        else:
                model.add(Dense(hp_units_1, input_shape=(2,), 
                        activation=activation)) 

        if activation == "RBF":
                model.add(RBFLayer(hp_units_2,
                        hp.Float("mu", min_value=1e-4, max_value=1),
                        input_shape=(2,)))
        else:
                model.add(Dense(hp_units_2, input_shape=(2,), 
                        activation=activation)) 
        
        model.add(Dense(1, activation='sigmoid'))

        if optimizer_choice == "Adam":
                optimizer = optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_choice == "Adagrad":
                optimizer = optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_choice == "SGD":
                optimizer = optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_choice == "RMSprop":
                optimizer = optimizers.RMSprop(learning_rate=learning_rate)
        else:
                raise ValueError
                    
        model.compile(optimizer=optimizer, 
                loss=losses,
                metrics=['accuracy'])
        return model

<<<<<<< HEAD
path = "calibrate_qubit_states/data.csv"
    
def classify_qubit(qubit, save_dir=pathlib.Path.cwd()):
=======
path = "data/calibrate_qubit_states/data.csv"
save_dir = pathlib.Path.cwd() 

for qubit in range(1,6):
>>>>>>> b7d4240d55d33ddfc45271aa4679bafff85737ac
    qubit_dir = save_dir / f"qubit{qubit}"
    qubit_dir.mkdir()

    data = pd.read_csv(path,skiprows=[1])
    data = data[data.qubit==qubit]
    print("qubit: ",qubit)
    print("size: ",len(data))
    print(data.head())

    f, axes = plt.subplots(1, 2,figsize=(14,7))
    sns.scatterplot(x='i', y='q', data=data, hue='state', ec=None, ax=axes[0],s=1)
    sns.countplot(x=data.state, data=data, ax=axes[1] )
    axes[1].set_title("states distribution")
    plt.savefig(qubit_dir / "data_processing.pdf")

    #shuffle dataset
    data = data.sample(frac =1)
    input_data = data[["i","q"]].values*10000 # WARNINGchange unit measure
    output_data = data["state"].values 
    # Split data into X_train, X_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.25, random_state= 0)

    tuner = kt.Hyperband(model_builder,
                        objective='val_accuracy',
<<<<<<< HEAD
                        max_epochs=150,
                        directory = qubit_dir, 
                        project_name = "NNmodel"
=======
                        max_epochs=150
>>>>>>> b7d4240d55d33ddfc45271aa4679bafff85737ac
                        )
    tuner.search_space_summary()

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

    tuner.search(x_train, y_train, epochs=120, validation_split=0.2,callbacks = [stop_early])

    # Get the optimal hyperparameters
    best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

    print(best_hps)

    neural_network= tuner.get_best_models()[0]
    hp_name = ['units_1','units_2', 'activation', 'optimizer', 'learning_rate','losses','add_normalisation']
    for i in hp_name:
        print(i,": ",best_hps.get(i))

    model = tuner.hypermodel.build(best_hps)
    epochs = 200
    start = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)
    stop = time.time()
    training_time = stop - start
    print("training time: ",training_time)

    history_dict = history.history
    results_model=pd.DataFrame(history_dict)

    plt.figure(figsize=(14,7))
    plt.plot( range(epochs),history_dict["loss"],label="loss")
    plt.plot( range(epochs),history_dict["accuracy"],label='accurancy')
    plt.plot( range(epochs),history_dict["val_loss"],label='val_loss')
    plt.plot( range(epochs),history_dict["val_accuracy"],label='val_accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(qubit_dir / "NN_training.pdf")

    start = time.time()
    loss_and_metrics = neural_network.evaluate(x_test, y_test)
    stop = time.time()
    classification_time = stop - start 
    print("NN trainig")
    print("classification time per item:", classification_time/len(x_test))
    print('Loss = ',loss_and_metrics[0])
    print('Accuracy = ',loss_and_metrics[1])

    y_pred = np.round(neural_network.predict(x_test))
    confusion_matrices = []
    confusion_matrices.append(confusion_matrix(y_test,y_pred,normalize="true"))

    ml_results = {
        "model":["Neural Network"],
        "accuracy" : [loss_and_metrics[1]],
        "testing time" : [classification_time/len(x_test)],
        "training time" : [training_time]
    }

    # RBF SVM

    print("RBF SVM")

    clf = SVC(gamma='auto')

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space['C'] = np.linspace(0.01,2,num = 50)
    space['degree'] = [2,3,4]
    search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=cv)
    result = search.fit(x_train, y_train)

    svm =  make_pipeline(StandardScaler(), search.best_estimator_)
    confusion_matrices.append(results(svm,x_train,y_train,x_test,y_test,ml_results,"RBF SVM"))

    # NAIVE BAYES

    print("NAIVE BAYES")

    naive_bayes =  make_pipeline(StandardScaler(), GaussianNB())
    confusion_matrices.append(results(naive_bayes,x_train,y_train,x_test,y_test,ml_results, 'Naive Bayes'))

    # LINEAR SVM

    print('LINEAR SVM')

    linear_svm =  make_pipeline(StandardScaler(), SVC(kernel="linear", C=0.025))
    confusion_matrices.append(results(linear_svm,x_train,y_train,x_test,y_test,ml_results, 'Linear SVM'))

    # GAUSSIAN PROCESS

    print('GAUSSIAN PROCESS')

    gaussian =  make_pipeline(StandardScaler(), GaussianProcessClassifier(1.0 * RBF(1.0)))
    confusion_matrices.append(results(gaussian,x_train,y_train,x_test,y_test,ml_results, 'Gaussian Process'))

    # RANDOM FOREST 

    print('RANDOM FOREST')

    clf = RandomForestClassifier()

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space['n_estimators'] = np.linspace(10,200,num = 20).astype('int')
    space['criterion'] = ['gini','entropy','log_loss']
    space['max_features'] = ['sqrt','log2',None]
    search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=cv)
    result = search.fit(x_train, y_train)

    random_forest =  make_pipeline(StandardScaler(), search.best_estimator_)
    confusion_matrices.append(results(random_forest,x_train,y_train,x_test,y_test,ml_results, 'Random Forest'))

    # ADA BOOST 

    print('ADA BOOST')

    clf = AdaBoostClassifier()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    space = dict()
    space['n_estimators'] = np.linspace(10,200,num = 20).astype('int')
    space['learning_rate'] = np.linspace(0.1,1,num = 10)
    space['algorithm'] = ['SAMME', 'SAMME.R']
    search = GridSearchCV(clf, space, scoring='accuracy', n_jobs=-1, cv=cv)
    result = search.fit(x_train, y_train)

    ada_boost =  make_pipeline(StandardScaler(), search.best_estimator_)
    confusion_matrices.append(results(ada_boost,x_train,y_train,x_test,y_test,ml_results, 'Ada Boost'))

    ml_results_pd = pd.DataFrame(ml_results)
    ml_results_pd.to_csv(qubit_dir / "results.csv")
    ml_results_pd["testing time"] *=1e5

    sns.set_style("darkgrid")
    g = sns.PairGrid(ml_results_pd, y_vars="model", 
                    x_vars=["accuracy", "testing time", "training time"], 
                    height=4, hue ='model', palette = "bright")
    g.map(sns.scatterplot)
<<<<<<< HEAD
    plt.xscale('log')
=======
>>>>>>> b7d4240d55d33ddfc45271aa4679bafff85737ac
    plt.savefig(qubit_dir / "benchmarks.pdf")

    from sklearn.metrics import RocCurveDisplay
    nn = KerasClassifier(neural_network)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    models=[nn,svm, naive_bayes,linear_svm, gaussian,random_forest, ada_boost]
    models_name = [ "Neural Network", "RBF SVM", "Naive Bayes","Linear SVM", "Gaussian Process",'Random Forest','Ada Boost' ]
    i = 1
    figure = plt.figure(figsize=(20,8))
    for name, clf in zip(models_name, models):
            
            ax = plt.subplot(1, len(models_name), i)
            clf.fit(x_train, y_train)
            score = clf.score(x_test, y_test)
            DecisionBoundaryDisplay.from_estimator(
                clf, x_train, cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )
            ax.scatter(
                x_test[:, 0],
                x_test[:, 1],
                c=y_test,
                cmap=cm_bright,
                edgecolors="k",
                
            )

            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            i += 1
    plt.savefig(qubit_dir / "results.pdf")

    figure = plt.figure(figsize=(30,5))
    for i, conf_matrix in enumerate( confusion_matrices):
        ax = plt.subplot(1, len(models_name), i+1)
        sns.heatmap(conf_matrix, annot = True,xticklabels=["P","N"],yticklabels=["P","N"])
        ax.set_title(models_name[i])

    plt.savefig(qubit_dir / "confusion_matrices.pdf")
    
    figure = plt.figure(figsize=(30,5))
    i=0
    for name, clf in zip(models_name, models):
        ax = plt.subplot(1, len(models_name), i+1)
        plt.subplot(1, len(models_name), i+1)
        RocCurveDisplay.from_estimator(clf, x_test, y_test,ax=ax,color="darkorange",)
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{name}")
        plt.legend()
        i+=1

    plt.savefig(qubit_dir / "ROC_curves.pdf")
<<<<<<< HEAD

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()

    save_dir = pathlib.Path.cwd() / f"_{args.folder}"
    save_dir.mkdir()

    for qubit in range(1,6):
        classify_qubit(qubit, save_dir=save_dir)
=======
>>>>>>> b7d4240d55d33ddfc45271aa4679bafff85737ac
