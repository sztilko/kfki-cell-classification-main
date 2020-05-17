from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
import numpy as np
import pandas as pd


def create_train_test_data(data_path, test_split=0.1):
  data = pd.read_csv(data_path,header=None)
  time = data.iloc[0,1:].astype('int') #timesteps in seconds if needed
  data.drop(data.index[0],inplace=True)
  test_data = data.sample(frac=test_split, random_state=42)
  train_data = data.drop(test_data.index)
  train_data = train_data.sample(frac=1,random_state=42).reset_index(drop=True)
  train_data.to_csv('train.csv')
  print(f'Train data samples: {train_data.shape[0]}')
  test_data.to_csv('test.csv')
  print(f'Test data samples: {test_data.shape[0]}')

def preprocess_data(data_path, normalize=False):
    data = pd.read_csv(data_path, index_col=0).reset_index(drop=True)
    label = data.pop('0')
    data_x = np.expand_dims(np.asarray(data).astype('float32'), -1)  # create compatible format with channel number=1 for Keras input
    if normalize:
        data_x = data_x / 1500
    label_encoder = LabelEncoder()
    label_int = label_encoder.fit_transform(label)  # convert to int labels
    data_y = tf.keras.utils.to_categorical(label_int, num_classes=2)  # convert to one-hot labels

    return data_x, data_y

