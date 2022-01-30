import tensorflow as tf
import time
import os


def creating_model(LOSS_FUNCTION,OPTIMIZER,METRICS,NUM_CLASSES):

    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28],name="inputLayer"),
          tf.keras.layers.Dense(300,activation='relu',name="HiddenLayer1"),
          tf.keras.layers.Dense(200,activation='relu',name="HiddenLayer2"),
          tf.keras.layers.Dense(100,activation='relu',name="HiddenLayer3"),
          tf.keras.layers.Dense(NUM_CLASSES,activation='softmax',name="OutputLayer"),
          
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()

    LOSS_FUNCTION = LOSS_FUNCTION
    OPTIMIZER = OPTIMIZER
    METRICS = METRICS

    model_clf.compile(loss=LOSS_FUNCTION , optimizer=OPTIMIZER ,metrics=METRICS)

    return model_clf

def get_unique_filename(filename):
    unique_fielname = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_fielname

def save_model(model , model_name , model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    model.save(path_to_model)

