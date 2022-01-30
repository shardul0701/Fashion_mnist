import tensorflow as tf

def get_data(validation):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train , y_train) , (X_test , y_test) = fashion_mnist.load_data()
    
    (X_val , X_train) = X_train[:5000]/255. , X_train[5000:]/255.
    (y_val , y_train) = y_train[:5000] , y_train[5000:]

    X_test = X_test/255

    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)

    return (X_train, y_train), (X_val, y_val) , (X_test, y_test)