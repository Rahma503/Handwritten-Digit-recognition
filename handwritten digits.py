#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# In[35]:


(X_train,y_train),(X_test,y_test)=keras.datasets.mnist.load_data()


# In[36]:


plt.matshow(X_train[6])


# In[37]:


y_train[6]


# In[38]:


X_train.shape


# In[39]:


X_train=X_train/255
X_test=X_test/255


# In[40]:


X_train_flat=X_train.reshape(len(X_train),784)
X_test_flat=X_test.reshape(len(X_test),784)


# In[41]:


X_test_flat.shape


# In[42]:


model=keras.Sequential([
    keras.layers.Dense(100,input_shape=(784,),activation='sigmoid')   
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',
            metrics=["accuracy"]
)


# In[43]:


model.fit(X_train_flat,y_train,epochs=5)


# In[44]:


loss,accuracy=model.evaluate(X_test_flat,y_test)
print("accuracy = ",accuracy)


# In[45]:


plt.matshow(X_test[9])


# In[46]:


y_pred=model.predict(X_test_flat)


# In[47]:


np.argmax(y_pred[9])


# In[48]:


y_test[6:12]


# In[49]:


y_pred_labels= [np.argmax(i) for i in y_pred]
y_pred_labels[6:12]


# In[50]:


cm=tf.math.confusion_matrix(labels=y_test,predictions=y_pred_labels)
cm


# In[51]:


import seaborn as sn 
plt.figure(figsize=(10,7))


# In[52]:


sn.heatmap(cm,annot=True,fmt='d')
plt.xlabel('predicted')
plt.ylabel('actual')

