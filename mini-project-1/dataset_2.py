#!/usr/bin/env python
# coding: utf-8

# # Deep Feature Dataset

# ## Importing Deep Feature Dataset

# In[97]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score,confusion_matrix,ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer


# In[98]:


# Load the dataset
data = np.load('datasets/train/train_feature.npz', allow_pickle=True)
train_deep_X = data['features']
train_deep_Y = data['label']

# Load validation set
valid_data = np.load('datasets/valid/valid_feature.npz', allow_pickle=True)
valid_deep_X = valid_data['features']
valid_deep_Y = valid_data['label']


# ## Taking % of training data set

# In[99]:


train_deep_X_80, train_deep_X_20, train_deep_Y_80, train_deep_Y_20 = train_test_split(train_deep_X, train_deep_Y, test_size=0.2, stratify=train_deep_Y, random_state=42)
train_deep_X_60, train_deep_X_40, train_deep_Y_60, train_deep_Y_40 = train_test_split(train_deep_X, train_deep_Y, test_size=0.4, stratify=train_deep_Y, random_state=42)
train_deep_X_100 = train_deep_X
train_deep_Y_100 = train_deep_Y


# ## For 100% Training Data

# ### Feature Transformation

# In[100]:


train_X_deep_flattened_100 = train_deep_X_100.reshape(train_deep_X_100.shape[0], -1)
valid_X_deep_flattened_100 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_100.shape)


# In[101]:


trf_100= FunctionTransformer(func=np.log1p)

train_X_deep_flattened_100 = trf_100.fit_transform(train_X_deep_flattened_100)
valid_X_deep_flattened_100 = trf_100.transform(valid_X_deep_flattened_100)


# ### Feature Reduction

# In[102]:


pca_100 = PCA(n_components=100)
train_X_deep_flattened_100 = pca_100.fit_transform(train_X_deep_flattened_100)
valid_X_deep_flattened_100 = pca_100.transform(valid_X_deep_flattened_100)
print(train_X_deep_flattened_100.shape)


# ### Model training

# In[103]:


model_100=SVC(C=100, degree=2, gamma='auto', kernel='rbf')

model_100.fit(train_X_deep_flattened_100, train_deep_Y_100)

y_pred_train_100 = model_100.predict(train_X_deep_flattened_100)

y_pred_valid_100 = model_100.predict(valid_X_deep_flattened_100)


# ### Accuracy

# In[104]:


accuracy_100 = accuracy_score(valid_deep_Y, y_pred_valid_100)
conf_matrix = confusion_matrix(valid_deep_Y, y_pred_valid_100)

print(f"Validation Accuracy: {accuracy_100*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 80% Training Data

# ### Feature Transformation

# In[105]:


train_X_deep_flattened_80 = train_deep_X_80.reshape(train_deep_X_80.shape[0], -1)
valid_X_deep_flattened_80 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_80.shape)


# In[106]:


trf_80= FunctionTransformer(func=np.log1p)

train_X_deep_flattened_80 = trf_80.fit_transform(train_X_deep_flattened_80)
valid_X_deep_flattened_80 = trf_80.transform(valid_X_deep_flattened_80)


# ### Feature Reduction

# In[107]:


pca_80 = PCA(n_components=100)
train_X_deep_flattened_80 = pca_80.fit_transform(train_X_deep_flattened_80)
valid_X_deep_flattened_80 = pca_80.transform(valid_X_deep_flattened_80)
print(train_X_deep_flattened_80.shape)


# ### Model Training

# In[108]:


model_80=SVC(C=100, degree=2, gamma='auto', kernel='rbf')

model_80.fit(train_X_deep_flattened_80, train_deep_Y_80)

y_pred_train_80 = model_80.predict(train_X_deep_flattened_80)

y_pred_valid_80 = model_80.predict(valid_X_deep_flattened_80)


# ### Accuracy

# In[109]:


accuracy_80 = accuracy_score(valid_deep_Y, y_pred_valid_80)
conf_matrix = confusion_matrix(valid_deep_Y, y_pred_valid_80)

print(f"Validation Accuracy: {accuracy_80*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 60% of Training Data

# ### Feature Transformation

# In[110]:


train_X_deep_flattened_60 = train_deep_X_60.reshape(train_deep_X_60.shape[0], -1)
valid_X_deep_flattened_60 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_60.shape)


# In[111]:


trf_60= FunctionTransformer(func=np.log1p)

train_X_deep_flattened_60 = trf_60.fit_transform(train_X_deep_flattened_60)
valid_X_deep_flattened_60 = trf_60.transform(valid_X_deep_flattened_60)


# ### Feature Reduction

# In[112]:


pca_60 = PCA(n_components=100)
train_X_deep_flattened_60 = pca_60.fit_transform(train_X_deep_flattened_60)
valid_X_deep_flattened_60 = pca_60.transform(valid_X_deep_flattened_60)
print(train_X_deep_flattened_60.shape)


# ### Model Training

# In[113]:


model_60=SVC(C=100, degree=2, gamma='auto', kernel='rbf')

model_60.fit(train_X_deep_flattened_60, train_deep_Y_60)

y_pred_train_60 = model_60.predict(train_X_deep_flattened_60)

y_pred_valid_60 = model_60.predict(valid_X_deep_flattened_60)


# ### Accuracy

# In[114]:


accuracy_60 = accuracy_score(valid_deep_Y, y_pred_valid_60)
conf_matrix = confusion_matrix(valid_deep_Y, y_pred_valid_60)

print(f"Validation Accuracy: {accuracy_60*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 40% Training Data

# ### Feature transformation

# In[115]:


train_X_deep_flattened_40 = train_deep_X_40.reshape(train_deep_X_40.shape[0], -1)
valid_X_deep_flattened_40 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_40.shape)


# In[116]:


trf_40= FunctionTransformer(func=np.log1p)

train_X_deep_flattened_40 = trf_40.fit_transform(train_X_deep_flattened_40)
valid_X_deep_flattened_40 = trf_40.transform(valid_X_deep_flattened_40)


# ### Feature Reduction

# In[117]:


pca_40 = PCA(n_components=100)
train_X_deep_flattened_40 = pca_40.fit_transform(train_X_deep_flattened_40)
valid_X_deep_flattened_40 = pca_40.transform(valid_X_deep_flattened_40)
print(train_X_deep_flattened_40.shape)


# ### Model Training

# In[118]:


model_40=SVC(C=100, degree=2, gamma='auto', kernel='rbf')

model_40.fit(train_X_deep_flattened_40, train_deep_Y_40)

y_pred_train_40 = model_40.predict(train_X_deep_flattened_40)

y_pred_valid_40 = model_40.predict(valid_X_deep_flattened_40)


# ### Accuracy

# In[119]:


accuracy_40 = accuracy_score(valid_deep_Y, y_pred_valid_40)
conf_matrix = confusion_matrix(valid_deep_Y, y_pred_valid_40)

print(f"Validation Accuracy: {accuracy_40*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 20% of Training Data

# ### Feature Transformation

# In[120]:


train_X_deep_flattened_20 = train_deep_X_20.reshape(train_deep_X_20.shape[0], -1)
valid_X_deep_flattened_20 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_20.shape)


# In[121]:


trf_20= FunctionTransformer(func=np.log1p)

train_X_deep_flattened_20 = trf_20.fit_transform(train_X_deep_flattened_20)
valid_X_deep_flattened_20 = trf_20.transform(valid_X_deep_flattened_20)


# ### Feature Reduction

# In[122]:


pca_20 = PCA(n_components=100)
train_X_deep_flattened_20 = pca_20.fit_transform(train_X_deep_flattened_20)
valid_X_deep_flattened_20 = pca_20.transform(valid_X_deep_flattened_20)
print(train_X_deep_flattened_20.shape)


# ### Model Training

# In[123]:


model_20=SVC(C=100, degree=2, gamma='auto', kernel='rbf')

model_20.fit(train_X_deep_flattened_20, train_deep_Y_20)

y_pred_train_20 = model_20.predict(train_X_deep_flattened_20)

y_pred_valid_20 = model_20.predict(valid_X_deep_flattened_20)


# ### Accuracy

# In[124]:


accuracy_20 = accuracy_score(valid_deep_Y, y_pred_valid_20)
conf_matrix = confusion_matrix(valid_deep_Y, y_pred_valid_20)

print(f"Validation Accuracy: {accuracy_20*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## Accuracy Variation plot for different % of data

# In[127]:


accuracy_scores = [accuracy_20*100, accuracy_40*100, accuracy_60*100, accuracy_80*100, accuracy_100*100]
percentage_of_data = [20, 40, 60, 80, 100]

plt.plot(percentage_of_data, accuracy_scores, color='red', marker='o')

plt.title('Deep Feature Dataset accuracies across different percentage of Training Data')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy Scores')

plt.ylim([95, 100])

plt.grid(True)
plt.show()


# ## Test Dataset Prediction

# In[126]:


test_data = np.load('datasets/test/test_feature.npz', allow_pickle=True)
test_deep_X = test_data['features']

test_X_deep_flattened = test_deep_X.reshape(test_deep_X.shape[0], -1)
test_X_deep_flattened = trf_100.transform(test_X_deep_flattened)
test_X_deep_flattened = pca_100.transform(test_X_deep_flattened)

y_pred_test = model_100.predict(test_X_deep_flattened)

np.savetxt("pred_deepfeat.txt", y_pred_test, fmt="%d", delimiter="\n")

