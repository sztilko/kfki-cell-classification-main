import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

class Classifier_CNN:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            # self.model.save_weights(self.output_directory + 'model_init.hdf5')
        return

    def build_model(self, input_shape, nb_classes):

        input_layer = tf.keras.layers.Input(input_shape)
        conv1 = tf.keras.layers.Conv1D(filters=6,kernel_size=7,activation='relu')(input_layer)
        conv2 = tf.keras.layers.Conv1D(filters=6,kernel_size=7,activation='relu')(conv1)
        gap = tf.keras.layers.GlobalAveragePooling1D()(conv2)
        output_layer = tf.keras.layers.Dense(units=nb_classes,activation='softmax')(gap)
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, batch_size, nb_epochs):

        hist = self.model.fit(x_train, y_train,
                              batch_size=batch_size,
                              epochs=nb_epochs,
                              verbose=self.verbose,
                              validation_data=(x_val, y_val))

        tf.keras.backend.clear_session()

    def predict(self, x_test,y_test,return_df_metrics = True):
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            return y_pred
    
    def evaluate(self, x_test,y_test):
        y_pred = self.model.predict(x_test) #get raw predictions
        y_pred_int = np.argmax(y_pred,axis=1) #transform predictions int encoding
        y_test_int = np.argmax(y_test,axis=1) 
        acc = sum(y_pred_int==y_test_int)/len(y_test_int)
        print(f'Test accuracy:\t{acc:.3} \n')

        return acc
    
    def save_model():
      pass
