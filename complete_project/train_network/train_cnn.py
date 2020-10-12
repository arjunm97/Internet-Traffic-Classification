#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.optimizers import Adam
from keras import optimizers
from keras.layers import Conv1D
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout


# In[36]:


# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)


# In[37]:


dataframe = pandas.read_csv("ITC_Dataset_latest.csv", header=1, na_values=["?"], usecols=range(0, 249), dtype = object)

#dataframe = pandas.read_csv("ITC_Dataset.csv", header=1) #itc_248.csv working


# In[38]:


#dataframe = dataframe.fillna(dataframe.mean())
dataframe = dataframe.fillna(0)


# In[ ]:





# In[ ]:





# In[39]:


dataset = dataframe.values


# In[40]:


#import numpy as np
#dataframe = dataframe.replace(numpy.nan, '', regex=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


X = dataset[:,0:248].astype(float)        #.astype(float)


# In[42]:


#X


# In[43]:


# load dataset
Y_ = dataset[:,248]


# In[44]:


Y_ = Y_.reshape(-1, 1) # Convert data to a single column


# In[ ]:





# In[45]:



# One Hot encode the class labels
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
Y = encoder.fit_transform(Y_)
encoder.categories_
#print(y)


# In[ ]:





# In[46]:


import sklearn
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.15)


# In[47]:


import keras


# In[49]:


seed = 42
numpy.random.seed(seed)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []


for train, test in kfold.split(X, Y_):
  # create model
	model = Sequential()
#at dense 50 got 78% using adam
	model.add(Dense(50, input_dim=248, activation='relu', name='fc1')) #input_shape=(248,)
	model.add(Dense(50, activation='relu', name='fc2'))
	#model.add(Dense(50, activation='relu', name='fc3'))
	model.add(Dropout(0.25))
	model.add(Dense(10, activation='softmax', name='output'))
	optimizer = Adam(lr=0.00005)
	#optimizer = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
	model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
	# Fit the model
    #Y = encoder.fit_transform(Y_)
	model.fit(X[train], Y[train], validation_data = (X[test], Y[test]), verbose=1, epochs=400)
	# evaluate the model
	scores = model.evaluate(X[test], Y[test], verbose=1)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
model.save("kfold_latest_db_200ep.h5")


# In[ ]:


print('Neural Network Model Summary: ')
print(model.summary())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:












# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




