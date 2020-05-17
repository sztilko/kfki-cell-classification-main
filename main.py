import utils
import os
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from cnn import Classifier_CNN

cwd = os.getcwd()
data_path = os.path.join(cwd,'maxWS_data.csv')
utils.create_train_test_data(data_path)

train_x, train_y = utils.preprocess_data('train.csv')
test_x, test_y = utils.preprocess_data('test.csv')
print(train_y)

kf = KFold(n_splits=3)

print(train_x.shape)

accuracies = []
for train_index, validation_index in kf.split(train_x):

  train_x_kf, val_x_kf = train_x[train_index], train_x[validation_index]
  train_y_kf, val_y_kf = train_y[train_index], train_y[validation_index]

  #create classifier instance
  cnn_classifier = Classifier_CNN(cwd,train_x.shape[1:],2,verbose=False)

  accuracies.append(cnn_classifier.evaluate(val_x_kf,val_y_kf))

print(np.mean(accuracies))
# ------------------------------------------------

# y_pred = cnn_classifier.model.predict(test_x) #get raw predictions

# y_pred_int = np.argmax(y_pred,axis=1) #transform predictions int encoding
# y_test_int = np.argmax(test_y,axis=1) 

# labels = ['HeLa','Preo']


# print(f'Test accuracy:\t{sum(y_pred_int==y_test_int)/len(y_test_int):.3} \n')

# conf_mat = confusion_matrix(y_test_int, y_pred_int)

# df_cm = pd.DataFrame(conf_mat, index = [label+'_true' for label in labels],
#                   columns = [label+'_pred' for label in labels])
# print(df_cm)