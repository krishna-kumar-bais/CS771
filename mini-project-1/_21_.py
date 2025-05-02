#!/usr/bin/env python
# coding: utf-8

# <center><h1> CS771 - Intro to ML (Autumn 2024): Mini-project 1 </h1></center>
#
# # Introduction
# This project involves training four binary classification models with three binary classification datasets, one for each dataset, and one for combined dataset. Each of these datasets represents the same machine learning task and was generated from the same raw dataset. The 3 datasets only differ in terms of features being used to represent each input from the original raw dataset. Each of these 3 datasets further consists of a training set, a validation set, and a test set
#
# # Installation
#
# ```
# pip install numpy
# pip install pandas
# pip install matplotlib
# pip install sklearn
# pip install tensorflow.keras
# ```
#
# # Emoticon Dataset
# This Dataset contains 13 emoticons for each sample
#
# ## Importing the Emoticon Dataset
# Now we will import the training, and validation set
#
# ## Taking % of Training Dataset
# Since we have to check the accuracies when training the model with 20%, 40%, 60%, 80%, and 100% of training data, we will split the training dataset into two ways, once in 20% and 80% and assign them to different variables, and into 40% and 60% and assign them to different variables, so we will have training datasets with 20%, 40%, 60%, 80%, 100% of training data.
#
# ## For 100% of Training Dataset
# We will do the entire feature engineering and model prediction for 100% of training set.
#
# ### Feature transformation and encoding
# This will be done for both training and validation set, but the training is done with training dataset.
# - **Transformation**: Since the dataset is in the form of 13 character string, where each character is an emoticon, we will now convert it to 13 featured string where each feature conatins the ord value of the emoticon.
#
# - **Encoding**: Now the feature is embeddded using a text based deep learning. After that the features are flattened.
#
# - **Standardisation**: This done so that single feature doesn't become too big or small in comparison.
#
# ### Model Training
# After the feature engineering, we will now train the model using Support Vector Machine Model, with hyperparameter tuning so that the model provides best accuracy with validation set. Here the model is to trained with training set. With the trained model we will now predict the labels for validation set.
#
# ### Accuracy Checking
# With the predicted labels, we will now check the accuracy with the true labels already provided for validation set. We will also display confusion matrix.
#
# ## For 80% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 80% of the training set.
#
# ## For 60% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 60% of the training set.
#
# ## For 40% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 40% of the training set.
#
# ## For 20% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 20% of the training set.
#
# ## Accuracy Variation plot for different % of data
# Now we will plot a line graph of the accuries of the model with different percentage of training set.
#
# ## Prediction for test data
# We will import test set and then we will apply the same feature engineering used for training and validation set, and use the model trained with training set to predict the test set labels.
#
# # Deep Feature Dataset
# This dataset a matrix of shape (13,768) for each sample.
#
# ## Importing the Deep feature Dataset
# Now we will import the training, and validation set
#
# ## Taking % of Training Dataset
# Since we have to check the accuracies when training the model with 20%, 40%, 60%, 80%, and 100% of training data, we will split the training dataset into two ways, once in 20% and 80% and assign them to different variables, and into 40% and 60% and assign them to different variables, so we will have training datasets with 20%, 40%, 60%, 80%, 100% of training data.
#
# ## For 100% of Training Dataset
# We will do the entire feature engineering and model prediction for 100% of training set.
#
# ### Feature transformation
# This will be done for both training and validation set, but the training is done with training dataset. Here first the feature are flattened and then Log Function transformer is applied to normalise the data.
#
# ### Feature Reduction
# This will be done for both training and validation set, but the training is done with training dataset. After feature transformation, number of features will be too large. so we reduced it using Principle Component Analysis.
#
# ### Model Training
# After the feature engineering, we will now train the model using Support Vector Machine Model, with hyperparameter tuning so that the model provides best accuracy with validation set. Here the model is to trained with training set. With the trained model we will now predict the labels for validation set.
#
# ### Accuracy Checking
# With the predicted labels, we will now check the accuracy with the true labels already provided for validation set. We will also display confusion matrix.
#
# ## For 80% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 80% of the training set.
#
# ## For 60% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 60% of the training set.
#
# ## For 40% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 40% of the training set.
#
# ## For 20% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 20% of the training set.
#
# ## Accuracy Variation plot for different % of data
# Now we will plot a line graph of the accuries of the model with different percentage of training set.
#
# ## Prediction for test data
# We will import test set and then we will apply the same feature engineering used for training and validation set, and use the model trained with training set to predict the test set labels.
#
# # Text Sequence Dataset
# In this Dataset each sample is of the form of a string of length 50, where each character is a digit.
#
# ## Importing the Text Sequence Dataset
# Now we will import the training, and validation set
#
# ## Taking % of Training Dataset
# Since we have to check the accuracies when training the model with 20%, 40%, 60%, 80%, and 100% of training data, we will split the training dataset into two ways, once in 20% and 80% and assign them to different variables, and into 40% and 60% and assign them to different variables, so we will have training datasets with 20%, 40%, 60%, 80%, 100% of training data.
#
# ## For 100% of Training Dataset
# We will do the entire feature engineering and model prediction for 100% of training set.
#
# ## Feature Transformation and encoding
# This will be done for both training and validation set, but the training is done with training dataset.
# - **Transformation**: We will transform the feature to resemble the emoticon dataset after its feature transformation, and this possible as we can have a bijective mapping between emoticon and text sequence dataset. This can done without using the emoticon dataset, by stricly observing anf following the rules of the pattern.
#
# - **Encoding**: Now the feature is embeddded using a text based deep learning. After that the features are flattened.
#
# - **Standardisation**: This done so that single feature doesn't become too big or small in comparison.
#
# ### Model Training
# After the feature engineering, we will now train the model using Support Vector Machine Model, with hyperparameter tuning so that the model provides best accuracy with validation set. Here the model is to trained with training set. With the trained model we will now predict the labels for validation set.
#
# ### Accuracy Checking
# With the predicted labels, we will now check the accuracy with the true labels already provided for validation set. We will also display confusion matrix.
#
# ## For 80% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 80% of the training set.
#
# ## For 60% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 60% of the training set.
#
# ## For 40% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 40% of the training set.
#
# ## For 20% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 20% of the training set.
#
# ## Accuracy Variation plot for different % of data
# Now we will plot a line graph of the accuries of the model with different percentage of training set.
#
# ## Prediction for test data
# We will import test set and then we will apply the same feature engineering used for training and validation set, and use the model trained with training set to predict the test set labels.
#
# # Combine Dataset
# In this we will only use emoticon and deep feature dataset, as emoticon and text sequence dataset are virtually the same, and no need for us to repeat it twice.
#
# ## Importing the combine dataset
# Now we will import the training, and validation set, for both emoticon and deep feature dataset.
#
# ## Taking % of Training Dataset
# Since we have to check the accuracies when training the model with 20%, 40%, 60%, 80%, and 100% of training data, we will split the training dataset into two ways, once in 20% and 80% and assign them to different variables, and into 40% and 60% and assign them to different variables, so we will have training datasets with 20%, 40%, 60%, 80%, 100% of training data.
#
# ## For 100% of Training Dataset
# We will do the entire feature engineering and model prediction for 100% of training set.
#
# ### Feature Transformation and Encoding
# We will do feature engineering separately for emoticon and deep feature dataset
# - **Emoticon Dataset:** The feature engineering here will be same as the one done before, no changes
# - **Deep feature dataset:** The feature engineering here will be same as the one done before, no changes
#
# ### Model Training
# After the feature engineering, we will now train the model using Ensemble or Stacking Model, for combining the dataset, get a meta feature and train a new random forst classifier model. This model is  then used to predict labels for the validation set.
#
# ### Accuracy Checking
# With the predicted labels, we will now check the accuracy with the true labels already provided for validation set. We will also display confusion matrix.
#
# ## For 80% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 80% of the training set.
#
# ## For 60% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 60% of the training set.
#
# ## For 40% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 40% of the training set.
#
# ## For 20% of Training Dataset
# All the feature engineering and model training will be same as the one done with 100% training data, except here we will only use 20% of the training set.
#
# ## Accuracy Variation plot for different % of data
# Now we will plot a line graph of the accuries of the model with different percentage of training set.
#
# ## Prediction for test data
# We will import test set and then we will apply the same feature engineering used for training and validation set, and use the model trained with training set to predict the test set labels.


#!/usr/bin/env python
# coding: utf-8

# # Emoticon Dataset

# ## Importing Emoticon Dataset

# In[88]:


import numpy as np
import pandas as pd

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score,confusion_matrix,ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[89]:


train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon']
train_emoticon_Y = train_emoticon_df['label'].to_numpy()

valid_emoticon_df=pd.read_csv("datasets/valid/valid_emoticon.csv")
valid_emoticon_X = valid_emoticon_df['input_emoticon']
valid_emoticon_Y = valid_emoticon_df['label'].to_numpy()


# ## Taking % of training data set

# In[90]:


train_emoticon_X_100 = train_emoticon_X
train_emoticon_Y_100 = train_emoticon_Y
train_emoticon_X_80, train_emoticon_X_20, train_emoticon_Y_80, train_emoticon_Y_20 = train_test_split(train_emoticon_X, train_emoticon_Y, test_size=0.2, stratify=train_emoticon_Y, random_state=42)
train_emoticon_X_60, train_emoticon_X_40, train_emoticon_Y_60, train_emoticon_Y_40 = train_test_split(train_emoticon_X, train_emoticon_Y, test_size=0.4, stratify=train_emoticon_Y, random_state=42)


# ## For 100% Training Data

# ### Feature Transformation and encoding

# #### Transformation

# In[91]:


train_emoticon_X_data_100 = [list(input_str) for input_str in train_emoticon_X_100]
train_emoticon_X_data_100 = pd.DataFrame(train_emoticon_X_data_100)
train_emoticon_X_data_100 = train_emoticon_X_data_100.map(ord)
train_emoticon_X_data_100 = train_emoticon_X_data_100.astype(str)
train_emoticon_X_data_100 = train_emoticon_X_data_100.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_100 = pd.DataFrame(train_emoticon_X_data_100, columns=['text'])


# In[92]:


valid_emoticon_X_data_100 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_100 = pd.DataFrame(valid_emoticon_X_data_100)
valid_emoticon_X_data_100 = valid_emoticon_X_data_100.map(ord)
valid_emoticon_X_data_100 = valid_emoticon_X_data_100.astype(str)
valid_emoticon_X_data_100 = valid_emoticon_X_data_100.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_100 = pd.DataFrame(valid_emoticon_X_data_100, columns=['text'])


# #### Embedding

# In[93]:


train_df = train_emoticon_X_data_100
valid_df = valid_emoticon_X_data_100

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_100 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_100):
    return df['tokens'].apply(lambda x: [vocab_dict_100[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_100)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_100)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_100) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_100 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_100.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_100.summary()

train_labels = train_emoticon_Y_100
model_1_100.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_100 = Model(inputs=model_1_100.input, outputs=model_1_100.get_layer("embedding_layer").output)

train_embeddings_100 = embedding_model_100.predict(train_padded)
valid_embeddings_100 = embedding_model_100.predict(valid_padded)


# In[94]:


train_emotioc_X_flattened_100 = train_embeddings_100.reshape(train_embeddings_100.shape[0], -1)
valid_emotioc_X_flattened_100 = valid_embeddings_100.reshape(valid_embeddings_100.shape[0], -1)

train_emoticon_X_encoded_100=pd.DataFrame(train_emotioc_X_flattened_100)
valid_emoticon_X_encoded_100=pd.DataFrame(valid_emotioc_X_flattened_100)


# #### Feature Standardization

# In[95]:


scaler_100 = StandardScaler()
train_emoticon_X_encoded_100 = scaler_100.fit_transform(train_emoticon_X_encoded_100)
valid_emoticon_X_encoded_100 = scaler_100.transform(valid_emoticon_X_encoded_100)


# ### Model Training

# In[96]:


model_100=SVC(C=0.1, degree=2, gamma='auto', kernel='rbf')

model_100.fit(train_emoticon_X_encoded_100, train_emoticon_Y_100)

y_pred_train_100 = model_100.predict(train_emoticon_X_encoded_100)
y_pred_valid_100 = model_100.predict(valid_emoticon_X_encoded_100)


# ### Accuracy Checking

# In[97]:


accuracy_100 = accuracy_score(valid_emoticon_Y, y_pred_valid_100)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_100)

print(f"Accuracy: {accuracy_100*100:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 80% Training Data

# ### Feature Transformation and encoding

# #### Transformation

# In[98]:


train_emoticon_X_data_80 = [list(input_str) for input_str in train_emoticon_X_80]
train_emoticon_X_data_80 = pd.DataFrame(train_emoticon_X_data_80)
train_emoticon_X_data_80 = train_emoticon_X_data_80.map(ord)
train_emoticon_X_data_80 = train_emoticon_X_data_80.astype(str)
train_emoticon_X_data_80 = train_emoticon_X_data_80.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_80 = pd.DataFrame(train_emoticon_X_data_80, columns=['text'])


# In[99]:


valid_emoticon_X_data_80 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_80 = pd.DataFrame(valid_emoticon_X_data_80)
valid_emoticon_X_data_80 = valid_emoticon_X_data_80.map(ord)
valid_emoticon_X_data_80 = valid_emoticon_X_data_80.astype(str)
valid_emoticon_X_data_80 = valid_emoticon_X_data_80.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_80 = pd.DataFrame(valid_emoticon_X_data_80, columns=['text'])


# #### Embedding

# In[100]:


train_df = train_emoticon_X_data_80
valid_df = valid_emoticon_X_data_80

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_80 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_80):
    return df['tokens'].apply(lambda x: [vocab_dict_80[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_80)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_80)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_80) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_80 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_80.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_80.summary()

train_labels = train_emoticon_Y_80
model_1_80.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_80 = Model(inputs=model_1_80.input, outputs=model_1_80.get_layer("embedding_layer").output)

train_embeddings_80 = embedding_model_80.predict(train_padded)
valid_embeddings_80 = embedding_model_80.predict(valid_padded)


# In[101]:


train_emotioc_X_flattened_80 = train_embeddings_80.reshape(train_embeddings_80.shape[0], -1)
valid_emotioc_X_flattened_80 = valid_embeddings_80.reshape(valid_embeddings_80.shape[0], -1)

train_emoticon_X_encoded_80=pd.DataFrame(train_emotioc_X_flattened_80)
valid_emoticon_X_encoded_80=pd.DataFrame(valid_emotioc_X_flattened_80)


# #### Feature Standardization

# In[102]:


scaler_80 = StandardScaler()
train_emoticon_X_encoded_80 = scaler_80.fit_transform(train_emoticon_X_encoded_80)
valid_emoticon_X_encoded_80 = scaler_80.transform(valid_emoticon_X_encoded_80)


# ### Model Training

# In[103]:


model_80=SVC(C=0.1, degree=2, gamma='auto', kernel='rbf')

model_80.fit(train_emoticon_X_encoded_80, train_emoticon_Y_80)

y_pred_train_80 = model_80.predict(train_emoticon_X_encoded_80)
y_pred_valid_80 = model_80.predict(valid_emoticon_X_encoded_80)


# ### Accuracy Checking

# In[104]:


accuracy_80 = accuracy_score(valid_emoticon_Y, y_pred_valid_80)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_80)

print(f"Accuracy: {accuracy_80*100:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 60% Training Data

# ### Feature Transformation and embedding

# #### Transformation

# In[105]:


train_emoticon_X_data_60 = [list(input_str) for input_str in train_emoticon_X_60]
train_emoticon_X_data_60 = pd.DataFrame(train_emoticon_X_data_60)
train_emoticon_X_data_60 = train_emoticon_X_data_60.map(ord)
train_emoticon_X_data_60 = train_emoticon_X_data_60.astype(str)
train_emoticon_X_data_60 = train_emoticon_X_data_60.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_60 = pd.DataFrame(train_emoticon_X_data_60, columns=['text'])


# In[106]:


valid_emoticon_X_data_60 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_60 = pd.DataFrame(valid_emoticon_X_data_60)
valid_emoticon_X_data_60 = valid_emoticon_X_data_60.map(ord)
valid_emoticon_X_data_60 = valid_emoticon_X_data_60.astype(str)
valid_emoticon_X_data_60 = valid_emoticon_X_data_60.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_60 = pd.DataFrame(valid_emoticon_X_data_60, columns=['text'])


# #### Embedding

# In[107]:


train_df = train_emoticon_X_data_60
valid_df = valid_emoticon_X_data_60

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_60 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_60):
    return df['tokens'].apply(lambda x: [vocab_dict_60[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_60)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_60)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_60) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_60 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_60.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_60.summary()

train_labels = train_emoticon_Y_60
model_1_60.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_60 = Model(inputs=model_1_60.input, outputs=model_1_60.get_layer("embedding_layer").output)

train_embeddings_60 = embedding_model_60.predict(train_padded)
valid_embeddings_60 = embedding_model_60.predict(valid_padded)


# In[108]:


train_emotioc_X_flattened_60 = train_embeddings_60.reshape(train_embeddings_60.shape[0], -1)
valid_emotioc_X_flattened_60 = valid_embeddings_60.reshape(valid_embeddings_60.shape[0], -1)

train_emoticon_X_encoded_60=pd.DataFrame(train_emotioc_X_flattened_60)
valid_emoticon_X_encoded_60=pd.DataFrame(valid_emotioc_X_flattened_60)


# #### Feature Standardization

# In[109]:


scaler_60 = StandardScaler()
train_emoticon_X_encoded_60 = scaler_60.fit_transform(train_emoticon_X_encoded_60)
valid_emoticon_X_encoded_60 = scaler_60.transform(valid_emoticon_X_encoded_60)


# ### Model Training

# In[110]:


model_60=SVC(C=0.1, degree=2, gamma='auto', kernel='rbf')

model_60.fit(train_emoticon_X_encoded_60, train_emoticon_Y_60)

y_pred_train_60 = model_60.predict(train_emoticon_X_encoded_60)
y_pred_valid_60 = model_60.predict(valid_emoticon_X_encoded_60)


# ### Accuracy Checking

# In[111]:


accuracy_60 = accuracy_score(valid_emoticon_Y, y_pred_valid_60)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_60)

print(f"Accuracy: {accuracy_60*100:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 40% Training Data

# ### Feature Transformation and Encoding

# #### Transformation

# In[112]:


train_emoticon_X_data_40 = [list(input_str) for input_str in train_emoticon_X_40]
train_emoticon_X_data_40 = pd.DataFrame(train_emoticon_X_data_40)
train_emoticon_X_data_40 = train_emoticon_X_data_40.map(ord)
train_emoticon_X_data_40 = train_emoticon_X_data_40.astype(str)
train_emoticon_X_data_40 = train_emoticon_X_data_40.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_40 = pd.DataFrame(train_emoticon_X_data_40, columns=['text'])


# In[113]:


valid_emoticon_X_data_40 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_40 = pd.DataFrame(valid_emoticon_X_data_40)
valid_emoticon_X_data_40 = valid_emoticon_X_data_40.map(ord)
valid_emoticon_X_data_40 = valid_emoticon_X_data_40.astype(str)
valid_emoticon_X_data_40 = valid_emoticon_X_data_40.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_40 = pd.DataFrame(valid_emoticon_X_data_40, columns=['text'])


# #### Embedding

# In[114]:


train_df = train_emoticon_X_data_40
valid_df = valid_emoticon_X_data_40

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_40 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_40):
    return df['tokens'].apply(lambda x: [vocab_dict_40[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_40)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_40)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_40) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_40 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_40.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_40.summary()

train_labels = train_emoticon_Y_40
model_1_40.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_40 = Model(inputs=model_1_40.input, outputs=model_1_40.get_layer("embedding_layer").output)

train_embeddings_40 = embedding_model_40.predict(train_padded)
valid_embeddings_40 = embedding_model_40.predict(valid_padded)


# In[115]:


train_emotioc_X_flattened_40 = train_embeddings_40.reshape(train_embeddings_40.shape[0], -1)
valid_emotioc_X_flattened_40 = valid_embeddings_40.reshape(valid_embeddings_40.shape[0], -1)

train_emoticon_X_encoded_40=pd.DataFrame(train_emotioc_X_flattened_40)
valid_emoticon_X_encoded_40=pd.DataFrame(valid_emotioc_X_flattened_40)


# #### Feature Standardization

# In[116]:


scaler_40 = StandardScaler()
train_emoticon_X_encoded_40 = scaler_40.fit_transform(train_emoticon_X_encoded_40)
valid_emoticon_X_encoded_40 = scaler_40.transform(valid_emoticon_X_encoded_40)


# ### Model Training

# In[117]:


model_40=SVC(C=0.1, degree=2, gamma='auto', kernel='rbf')

model_40.fit(train_emoticon_X_encoded_40, train_emoticon_Y_40)

y_pred_train_40 = model_40.predict(train_emoticon_X_encoded_40)
y_pred_valid_40 = model_40.predict(valid_emoticon_X_encoded_40)


# ### Accuracy Checking

# In[118]:


accuracy_40 = accuracy_score(valid_emoticon_Y, y_pred_valid_40)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_40)

print(f"Accuracy: {accuracy_40*100:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 20% Training Data

# ### Feature Transformation and Encoding

# #### Transformation

# In[119]:


train_emoticon_X_data_20 = [list(input_str) for input_str in train_emoticon_X_20]
train_emoticon_X_data_20 = pd.DataFrame(train_emoticon_X_data_20)
train_emoticon_X_data_20 = train_emoticon_X_data_20.map(ord)
train_emoticon_X_data_20 = train_emoticon_X_data_20.astype(str)
train_emoticon_X_data_20 = train_emoticon_X_data_20.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_20 = pd.DataFrame(train_emoticon_X_data_20, columns=['text'])


# In[120]:


valid_emoticon_X_data_20 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_20 = pd.DataFrame(valid_emoticon_X_data_20)
valid_emoticon_X_data_20 = valid_emoticon_X_data_20.map(ord)
valid_emoticon_X_data_20 = valid_emoticon_X_data_20.astype(str)
valid_emoticon_X_data_20 = valid_emoticon_X_data_20.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_20 = pd.DataFrame(valid_emoticon_X_data_20, columns=['text'])


# #### Embedding

# In[121]:


train_df = train_emoticon_X_data_20
valid_df = valid_emoticon_X_data_20

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_20 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_20):
    return df['tokens'].apply(lambda x: [vocab_dict_20[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_20)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_20)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_20) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_20 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_20.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_20.summary()

train_labels = train_emoticon_Y_20
model_1_20.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_20 = Model(inputs=model_1_20.input, outputs=model_1_20.get_layer("embedding_layer").output)

train_embeddings_20 = embedding_model_20.predict(train_padded)
valid_embeddings_20 = embedding_model_20.predict(valid_padded)


# In[122]:


train_emotioc_X_flattened_20 = train_embeddings_20.reshape(train_embeddings_20.shape[0], -1)
valid_emotioc_X_flattened_20 = valid_embeddings_20.reshape(valid_embeddings_20.shape[0], -1)

train_emoticon_X_encoded_20=pd.DataFrame(train_emotioc_X_flattened_20)
valid_emoticon_X_encoded_20=pd.DataFrame(valid_emotioc_X_flattened_20)


# #### Feature Standardization

# In[123]:


scaler_20 = StandardScaler()
train_emoticon_X_encoded_20 = scaler_20.fit_transform(train_emoticon_X_encoded_20)
valid_emoticon_X_encoded_20 = scaler_20.transform(valid_emoticon_X_encoded_20)


# ### Model Training

# In[124]:


model_20=SVC(C=0.1, degree=2, gamma='auto', kernel='rbf')

model_20.fit(train_emoticon_X_encoded_20, train_emoticon_Y_20)

y_pred_train_20 = model_20.predict(train_emoticon_X_encoded_20)
y_pred_valid_20 = model_20.predict(valid_emoticon_X_encoded_20)


# ### Accuracy Checking

# In[125]:


accuracy_20 = accuracy_score(valid_emoticon_Y, y_pred_valid_20)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_20)

print(f"Accuracy: {accuracy_20*100:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## Accuracy Variation plot for different % of data

# In[126]:


accuracy_scores = [accuracy_20*100, accuracy_40*100, accuracy_60*100, accuracy_80*100, accuracy_100*100]
percentage_of_data = [20, 40, 60, 80, 100]

plt.plot(percentage_of_data, accuracy_scores, color='red', marker='o')

plt.title('Emoticon Dataset accuracies across different percentage of Training Data')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy Scores')

# plt.ylim([91, 98])

plt.grid(True)
plt.show()


# ## Prediction For Test dataset

# In[127]:


test_emoticon_df=pd.read_csv("datasets/test/test_emoticon.csv")
test_emoticon_X = test_emoticon_df['input_emoticon']

test_emoticon_X_data = [list(input_str) for input_str in test_emoticon_X]
test_emoticon_X_data = pd.DataFrame(test_emoticon_X_data)
test_emoticon_X_data = test_emoticon_X_data.map(ord)
test_emoticon_X_data = test_emoticon_X_data.astype(str)
test_emoticon_X_data = test_emoticon_X_data.apply(lambda row: ' '.join(row.values), axis=1)
test_emoticon_X_data = pd.DataFrame(test_emoticon_X_data, columns=['text'])


# In[128]:


test_df = test_emoticon_X_data
test_df['tokens'] = test_df['text'].apply(lambda x: x.split())
def tokenize_test_data(df, vocab_dict_100):
    return df['tokens'].apply(lambda x: [vocab_dict_100[token] if token in vocab_dict_100 else 0 for token in x])
test_df['tokenized_text'] = tokenize_test_data(test_df, vocab_dict_100)
test_padded = pad_sequences(test_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
test_embeddings = embedding_model_100.predict(test_padded)
print("Test Embeddings Shape:", test_embeddings.shape)
test_emoticon_X_flattened = test_embeddings.reshape(test_embeddings.shape[0], -1)
test_emoticon_X_encoded=pd.DataFrame(test_emoticon_X_flattened)
test_emoticon_X_encoded = scaler_100.transform(test_emoticon_X_encoded)
y_pred_test = model_100.predict(test_emoticon_X_encoded)
np.savetxt("pred_emoticon.txt", y_pred_test, fmt="%d", delimiter="\n")

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

#!/usr/bin/env python
# coding: utf-8

# # Text Sequence Dataset

# ## Importing Text sequence Dataset

# In[57]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.svm import SVC
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score,confusion_matrix,ConfusionMatrixDisplay)
import matplotlib.pyplot as plt


# In[58]:


train_text_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")

valid_text_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")

known_substring_1 = {'15436': 'a','464': 'b', '422': 'c', '1596':'d', '614': 'e', '262': 'f', '284': 'g'}
known_substring_2 = {'a':'15436', 'b':'464', 'c':'422', 'd':'1586', 'e':'614', 'f':'262', 'g':'284'}


# ## Taking % of training data set

# In[59]:


size_train = len(train_text_seq_df)
size_valid = len(valid_text_seq_df)

train_text_seq_X = train_text_seq_df['input_str']
train_text_seq_Y = train_text_seq_df['label'].to_numpy()

valid_text_seq_X = valid_text_seq_df['input_str']
valid_text_seq_Y = valid_text_seq_df['label'].to_numpy()


# In[60]:


train_text_seq_X_100 = train_text_seq_X
train_text_seq_Y_100 = train_text_seq_Y
train_text_seq_X_80, train_text_seq_X_20, train_text_seq_Y_80, train_text_seq_Y_20 = train_test_split(train_text_seq_X, train_text_seq_Y, test_size=0.2, stratify=train_text_seq_Y, random_state=42)
train_text_seq_X_60, train_text_seq_X_40, train_text_seq_Y_60, train_text_seq_Y_40 = train_test_split(train_text_seq_X, train_text_seq_Y, test_size=0.4, stratify=train_text_seq_Y, random_state=42)


# In[61]:


valid_text_seq_X_100 = valid_text_seq_X
valid_text_seq_X_80 = valid_text_seq_X
valid_text_seq_X_60 = valid_text_seq_X
valid_text_seq_X_40 = valid_text_seq_X
valid_text_seq_X_20 = valid_text_seq_X


# In[62]:


valid_text_seq_Y_100 = valid_text_seq_Y
valid_text_seq_Y_80 = valid_text_seq_Y
valid_text_seq_Y_60 = valid_text_seq_Y
valid_text_seq_Y_40 = valid_text_seq_Y
valid_text_seq_Y_20 = valid_text_seq_Y


# In[63]:


size_train_100 = len(train_text_seq_X_100)
size_train_80 = len(train_text_seq_X_80)
size_train_60 = len(train_text_seq_X_60)
size_train_40 = len(train_text_seq_X_40)
size_train_20 = len(train_text_seq_X_20)


# ## For 100% Training Data

# ### Transforming Dataset set to a pattern similar to Emoticon Dataset

# #### Training Dataset Transformation

# In[64]:


train_text_seq_X_100 = train_text_seq_X_100.str.replace('15436', 'a')

train_text_seq_X_100=train_text_seq_X_100.to_numpy()
train_text_seq_X_100 = [[sample[:-17], sample[-17:]] for sample in train_text_seq_X_100]
train_text_seq_X_100 = pd.DataFrame(train_text_seq_X_100)

train_text_seq_X_100[1] = train_text_seq_X_100[1].str.replace('1596', 'd')
train_text_seq_X_100[1] = train_text_seq_X_100[1].str.replace('284', 'g')
train_text_seq_X_100[1] = train_text_seq_X_100[1].str.replace('614','e')

for i in range (0,size_train_100):
    if train_text_seq_X_100.iloc[i,1].count('262') != 1:
        n = len(train_text_seq_X_100.iloc[i,1])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_100.iloc[i,1][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2:g[j+2]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_100.iloc[i,1][j:j+3]=='262'):
                a=train_text_seq_X_100.iloc[i,1][:j]
                b=train_text_seq_X_100.iloc[i,1][j:j+3]
                c=train_text_seq_X_100.iloc[i,1][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_100.iloc[i,1]=a+b+c
                break
    else:
        train_text_seq_X_100.iloc[i,1] = train_text_seq_X_100.iloc[i,1].replace('262','f')

train_text_seq_X_100['input_str']=train_text_seq_X_100[0]+train_text_seq_X_100[1]
train_text_seq_X_100=train_text_seq_X_100.drop(columns=[0,1])

train_text_seq_X_100['input_str'] = train_text_seq_X_100['input_str'].str.replace('1596','d')
train_text_seq_X_100['input_str'] = train_text_seq_X_100['input_str'].str.lstrip('0')

for i in range (0,size_train_100):
    if train_text_seq_X_100.iloc[i,0].count('262') == 1 and train_text_seq_X_100.iloc[i,0].count('26262') == 0:
        train_text_seq_X_100.iloc[i,0] = train_text_seq_X_100.iloc[i,0].replace('262','f')
    if train_text_seq_X_100.iloc[i,0].count('464') == 1:
        train_text_seq_X_100.iloc[i,0] = train_text_seq_X_100.iloc[i,0].replace('464','b')
    if train_text_seq_X_100.iloc[i,0].count('422') == 1:
        train_text_seq_X_100.iloc[i,0] = train_text_seq_X_100.iloc[i,0].replace('422','c')
    if train_text_seq_X_100.iloc[i,0].count('614') == 1:
        train_text_seq_X_100.iloc[i,0] = train_text_seq_X_100.iloc[i,0].replace('614','e')

for i in range (0,size_train_100):
    if train_text_seq_X_100.iloc[i,0].count('262') == 1 and train_text_seq_X_100.iloc[i,0].count('26262') == 0 and train_text_seq_X_100.iloc[i,0].count('f')==1:
        train_text_seq_X_100.iloc[i,0] = train_text_seq_X_100.iloc[i,0].replace('262','f')
    if train_text_seq_X_100.iloc[i,0].count('464') == 1 and train_text_seq_X_100.iloc[i,0].count('b')==0:
        train_text_seq_X_100.iloc[i,0] = train_text_seq_X_100.iloc[i,0].replace('464','b')
    if train_text_seq_X_100.iloc[i,0].count('422') == 1 and train_text_seq_X_100.iloc[i,0].count('c')==0:
        train_text_seq_X_100.iloc[i,0] = train_text_seq_X_100.iloc[i,0].replace('422','c')
    if train_text_seq_X_100.iloc[i,0].count('614') == 1 and train_text_seq_X_100.iloc[i,0].count('e')==1:
        train_text_seq_X_100.iloc[i,0] = train_text_seq_X_100.iloc[i,0].replace('614','e')

for i in range (0,size_train_100):
    if train_text_seq_X_100.iloc[i,0].count('b') == 0:
        n = len(train_text_seq_X_100.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_100.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_100.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_100.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_100.iloc[i,0][j:j+3]=='464'):
                a=train_text_seq_X_100.iloc[i,0][:j]
                b=train_text_seq_X_100.iloc[i,0][j:j+3]
                c=train_text_seq_X_100.iloc[i,0][j+3:]
                b=b.replace('464','b')
                train_text_seq_X_100.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_100):
    if train_text_seq_X_100.iloc[i,0].count('c') == 0:
        n = len(train_text_seq_X_100.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_100.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_100.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_100.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_100.iloc[i,0][j:j+3]=='422'):
                a=train_text_seq_X_100.iloc[i,0][:j]
                b=train_text_seq_X_100.iloc[i,0][j:j+3]
                c=train_text_seq_X_100.iloc[i,0][j+3:]
                b=b.replace('422','c')
                train_text_seq_X_100.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_100):
    if train_text_seq_X_100.iloc[i,0].count('e') == 1:
        n = len(train_text_seq_X_100.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_100.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_100.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_100.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_100.iloc[i,0][j:j+3]=='614'):
                a=train_text_seq_X_100.iloc[i,0][:j]
                b=train_text_seq_X_100.iloc[i,0][j:j+3]
                c=train_text_seq_X_100.iloc[i,0][j+3:]
                b=b.replace('614','e')
                train_text_seq_X_100.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_100):
    if train_text_seq_X_100.iloc[i,0].count('f') == 1:
        n = len(train_text_seq_X_100.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_100.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_100.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_100.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_100.iloc[i,0][j:j+3]=='262'):
                a=train_text_seq_X_100.iloc[i,0][:j]
                b=train_text_seq_X_100.iloc[i,0][j:j+3]
                c=train_text_seq_X_100.iloc[i,0][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_100.iloc[i,0]=a+b+c
                break

for i in range(0,size_train_100):
    a=train_text_seq_X_100.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_100.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

for i in range(0,size_train_100):
    k=0
    check=False
    check1=False
    a=train_text_seq_X_100.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0:
                if l<3:
                    check=True
                    break
                else:
                    if l>5:
                        k+=2
                    else:
                        k+=1
                if k>3:
                    check1=True
                    break
                l=0
    if check:
        continue
    if l!=0:
        if l<3:
            check=True
        else:
            if l>5:
                k+=2
            else:
                k+=1
        if k>3:
            check1=True
    if check:
        continue
    if check1:
        if a.count('262')>0:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('262','temp')
            x=x.replace('f','262')
            x=x.replace('temp','f')
            a=x+y
            train_text_seq_X_100.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('422','temp')
            x=x.replace('c','422')
            x=x.replace('temp','c')
            a=x+y
            train_text_seq_X_100.iloc[i,0]=a

for i in range(0,size_train_100):
    check=False
    a=train_text_seq_X_100.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0 and l<3:
                check=True
            l=0
    if a[n-1].isdigit() and l!=0 and l<3:
        check=True
    if check:
        if a.count('262')==0:
            if a.count('614')>0:
                x=a[:-8]
                y=a[-8:]
                x=x.replace('614','nis')
                x=x.replace('e','614')
                x=x.replace('nis','e')
                a=x+y
            elif a.count('f62')>0:
                a=a.replace('f62','26f')
            else:
                a=a.replace('b64','46b')
            train_text_seq_X_100.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            if y.count('262')>0:
                y=y.replace('262','nis')
                y=y.replace('f','262')
                y=y.replace('nis','f')
            if x.count('262')>0:
                x=x.replace('262','nis')
                x=x.replace('f','262')
                x=x.replace('nis','f')
            a=x+y
            train_text_seq_X_100.iloc[i,0]=a

for i in range(0,size_train_100):
    a=train_text_seq_X_100.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_100.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

train_text_seq_X_transformed_100 = pd.DataFrame([[''] * 13] * size_train_100)
value_list_train = []

for i in range (0,size_train_100):
    a=train_text_seq_X_100.iloc[i,0]
    n=len(a)
    k=''
    r=0
    for j in range (0,n):
        if a[j].isdigit():
            k=k+a[j]
        else:
            if len(k)>5:
                if len(k)>10:
                    train_text_seq_X_transformed_100.iloc[i,r]=k
                    r+=3
                else:
                    train_text_seq_X_transformed_100.iloc[i,r]=k
                    r+=2
                train_text_seq_X_transformed_100.iloc[i,r]=a[j]
                r+=1
            else:
                if len(k)==0:
                    train_text_seq_X_transformed_100.iloc[i,r]=a[j]
                    r+=1
                else:
                    value_list_train.append(k)
                    train_text_seq_X_transformed_100.iloc[i,r]=k
                    r+=1
                    train_text_seq_X_transformed_100.iloc[i,r]=a[j]
                    r+=1
            k=''
    if r<13:
        value_list_train.append(k)
        train_text_seq_X_transformed_100.iloc[i,12]=k
unique_list_train=set(value_list_train)

for i in range (0,size_train_100):
    for j in range (0,13):
        a=train_text_seq_X_transformed_100.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_100.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_100.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_100.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_100.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_100.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_100.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_100.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_100.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_100.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_100.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_100.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_100.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_100):
    for j in range (0,13):
        a=train_text_seq_X_transformed_100.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_100.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_100.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_100.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_100.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_100.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_100.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_100.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_100.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_100.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_100.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_100.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_100.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_100):
    for j in range (0,13):
        a=train_text_seq_X_transformed_100.iloc[i,j]
        if a=='':
            b=train_text_seq_X_transformed_100.iloc[i,j-1]
            train_text_seq_X_transformed_100.iloc[i,j-1]=b[:4]
            train_text_seq_X_transformed_100.iloc[i,j]=b[4:]
            break


# #### Validation Dataset Transformation

# In[65]:


valid_text_seq_X_100 = valid_text_seq_X_100.str.replace('15436', 'a')

valid_text_seq_X_100=valid_text_seq_X_100.to_numpy()
valid_text_seq_X_100 = [[sample[:-17], sample[-17:]] for sample in valid_text_seq_X_100]
valid_text_seq_X_100 = pd.DataFrame(valid_text_seq_X_100)

valid_text_seq_X_100[1] = valid_text_seq_X_100[1].str.replace('1596', 'd')
valid_text_seq_X_100[1] = valid_text_seq_X_100[1].str.replace('284', 'g')
valid_text_seq_X_100[1] = valid_text_seq_X_100[1].str.replace('614','e')

for i in range (0,size_valid):
    if valid_text_seq_X_100.iloc[i,1].count('262') != 1:
        n = len(valid_text_seq_X_100.iloc[i,1])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not valid_text_seq_X_100.iloc[i,1][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2:g[j+2]=0
        for j in range (0,n):
            if g[j] and (valid_text_seq_X_100.iloc[i,1][j:j+3]=='262'):
                a=valid_text_seq_X_100.iloc[i,1][:j]
                b=valid_text_seq_X_100.iloc[i,1][j:j+3]
                c=valid_text_seq_X_100.iloc[i,1][j+3:]
                b=b.replace('262','f')
                valid_text_seq_X_100.iloc[i,1]=a+b+c
                break
    else:
        valid_text_seq_X_100.iloc[i,1] = valid_text_seq_X_100.iloc[i,1].replace('262','f')

valid_text_seq_X_100['input_str']=valid_text_seq_X_100[0]+valid_text_seq_X_100[1]
valid_text_seq_X_100=valid_text_seq_X_100.drop(columns=[0,1])

valid_text_seq_X_100['input_str'] = valid_text_seq_X_100['input_str'].str.replace('1596','d')
valid_text_seq_X_100['input_str'] = valid_text_seq_X_100['input_str'].str.lstrip('0')

for i in range (0,size_valid):
    if valid_text_seq_X_100.iloc[i,0].count('262') == 1 and valid_text_seq_X_100.iloc[i,0].count('26262') == 0:
        valid_text_seq_X_100.iloc[i,0] = valid_text_seq_X_100.iloc[i,0].replace('262','f')
    if valid_text_seq_X_100.iloc[i,0].count('464') == 1:
        valid_text_seq_X_100.iloc[i,0] = valid_text_seq_X_100.iloc[i,0].replace('464','b')
    if valid_text_seq_X_100.iloc[i,0].count('422') == 1:
        valid_text_seq_X_100.iloc[i,0] = valid_text_seq_X_100.iloc[i,0].replace('422','c')
    if valid_text_seq_X_100.iloc[i,0].count('614') == 1:
        valid_text_seq_X_100.iloc[i,0] = valid_text_seq_X_100.iloc[i,0].replace('614','e')

for i in range (0,size_valid):
    if valid_text_seq_X_100.iloc[i,0].count('262') == 1 and valid_text_seq_X_100.iloc[i,0].count('26262') == 0 and valid_text_seq_X_100.iloc[i,0].count('f')==1:
        valid_text_seq_X_100.iloc[i,0] = valid_text_seq_X_100.iloc[i,0].replace('262','f')
    if valid_text_seq_X_100.iloc[i,0].count('464') == 1 and valid_text_seq_X_100.iloc[i,0].count('b')==0:
        valid_text_seq_X_100.iloc[i,0] = valid_text_seq_X_100.iloc[i,0].replace('464','b')
    if valid_text_seq_X_100.iloc[i,0].count('422') == 1 and valid_text_seq_X_100.iloc[i,0].count('c')==0:
        valid_text_seq_X_100.iloc[i,0] = valid_text_seq_X_100.iloc[i,0].replace('422','c')
    if valid_text_seq_X_100.iloc[i,0].count('614') == 1 and valid_text_seq_X_100.iloc[i,0].count('e')==1:
        valid_text_seq_X_100.iloc[i,0] = valid_text_seq_X_100.iloc[i,0].replace('614','e')

for i in range (0,size_valid):
    if valid_text_seq_X_100.iloc[i,0].count('b') == 0:
        n = len(valid_text_seq_X_100.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not valid_text_seq_X_100.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and valid_text_seq_X_100.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and valid_text_seq_X_100.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (valid_text_seq_X_100.iloc[i,0][j:j+3]=='464'):
                a=valid_text_seq_X_100.iloc[i,0][:j]
                b=valid_text_seq_X_100.iloc[i,0][j:j+3]
                c=valid_text_seq_X_100.iloc[i,0][j+3:]
                b=b.replace('464','b')
                valid_text_seq_X_100.iloc[i,0]=a+b+c
                break

for i in range (0,size_valid):
    if valid_text_seq_X_100.iloc[i,0].count('c') == 0:
        n = len(valid_text_seq_X_100.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not valid_text_seq_X_100.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and valid_text_seq_X_100.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and valid_text_seq_X_100.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (valid_text_seq_X_100.iloc[i,0][j:j+3]=='422'):
                a=valid_text_seq_X_100.iloc[i,0][:j]
                b=valid_text_seq_X_100.iloc[i,0][j:j+3]
                c=valid_text_seq_X_100.iloc[i,0][j+3:]
                b=b.replace('422','c')
                valid_text_seq_X_100.iloc[i,0]=a+b+c
                break

for i in range (0,size_valid):
    if valid_text_seq_X_100.iloc[i,0].count('e') == 1:
        n = len(valid_text_seq_X_100.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not valid_text_seq_X_100.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and valid_text_seq_X_100.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and valid_text_seq_X_100.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (valid_text_seq_X_100.iloc[i,0][j:j+3]=='614'):
                a=valid_text_seq_X_100.iloc[i,0][:j]
                b=valid_text_seq_X_100.iloc[i,0][j:j+3]
                c=valid_text_seq_X_100.iloc[i,0][j+3:]
                b=b.replace('614','e')
                valid_text_seq_X_100.iloc[i,0]=a+b+c
                break

for i in range (0,size_valid):
    if valid_text_seq_X_100.iloc[i,0].count('f') == 1:
        n = len(valid_text_seq_X_100.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not valid_text_seq_X_100.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and valid_text_seq_X_100.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and valid_text_seq_X_100.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (valid_text_seq_X_100.iloc[i,0][j:j+3]=='262'):
                a=valid_text_seq_X_100.iloc[i,0][:j]
                b=valid_text_seq_X_100.iloc[i,0][j:j+3]
                c=valid_text_seq_X_100.iloc[i,0][j+3:]
                b=b.replace('262','f')
                valid_text_seq_X_100.iloc[i,0]=a+b+c
                break

for i in range(0,size_valid):
    a=valid_text_seq_X_100.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                valid_text_seq_X_100.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

for i in range(0,size_valid):
    k=0
    check=False
    check1=False
    a=valid_text_seq_X_100.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0:
                if l<3:
                    check=True
                    break
                else:
                    if l>5:
                        k+=2
                    else:
                        k+=1
                if k>3:
                    check1=True
                    break
                l=0
    if check:
        continue
    if l!=0:
        if l<3:
            check=True
        else:
            if l>5:
                k+=2
            else:
                k+=1
        if k>3:
            check1=True
    if check:
        continue
    if check1:
        if a.count('262')>0:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('262','temp')
            x=x.replace('f','262')
            x=x.replace('temp','f')
            a=x+y
            valid_text_seq_X_100.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('422','temp')
            x=x.replace('c','422')
            x=x.replace('temp','c')
            a=x+y
            valid_text_seq_X_100.iloc[i,0]=a

for i in range(0,size_valid):
    check=False
    a=valid_text_seq_X_100.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0 and l<3:
                check=True
            l=0
    if a[n-1].isdigit() and l!=0 and l<3:
        check=True
    if check:
        if a.count('262')==0:
            if a.count('614')>0:
                x=a[:-8]
                y=a[-8:]
                x=x.replace('614','nis')
                x=x.replace('e','614')
                x=x.replace('nis','e')
                a=x+y
            elif a.count('f62')>0:
                a=a.replace('f62','26f')
            else:
                a=a.replace('b64','46b')
            valid_text_seq_X_100.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            if y.count('262')>0:
                y=y.replace('262','nis')
                y=y.replace('f','262')
                y=y.replace('nis','f')
            if x.count('262')>0:
                x=x.replace('262','nis')
                x=x.replace('f','262')
                x=x.replace('nis','f')
            a=x+y
            valid_text_seq_X_100.iloc[i,0]=a

for i in range(0,size_valid):
    a=valid_text_seq_X_100.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                valid_text_seq_X_100.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

valid_text_seq_X_transformed_100 = pd.DataFrame([[''] * 13] * size_valid)
value_list_valid = []

for i in range (0,size_valid):
    a=valid_text_seq_X_100.iloc[i,0]
    n=len(a)
    k=''
    r=0
    for j in range (0,n):
        if a[j].isdigit():
            k=k+a[j]
        else:
            if len(k)>5:
                if len(k)>10:
                    valid_text_seq_X_transformed_100.iloc[i,r]=k
                    r+=3
                else:
                    valid_text_seq_X_transformed_100.iloc[i,r]=k
                    r+=2
                valid_text_seq_X_transformed_100.iloc[i,r]=a[j]
                r+=1
            else:
                if len(k)==0:
                    valid_text_seq_X_transformed_100.iloc[i,r]=a[j]
                    r+=1
                else:
                    value_list_valid.append(k)
                    valid_text_seq_X_transformed_100.iloc[i,r]=k
                    r+=1
                    valid_text_seq_X_transformed_100.iloc[i,r]=a[j]
                    r+=1
            k=''
    if r<13:
        value_list_valid.append(k)
        valid_text_seq_X_transformed_100.iloc[i,12]=k
unique_list_valid=set(value_list_valid)

for i in range (0,size_valid):
    for j in range (0,13):
        a=valid_text_seq_X_transformed_100.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_valid:
                    ok1=True
                    if a[5:] in unique_list_valid:
                        valid_text_seq_X_transformed_100.iloc[i,j]=a[:5]
                        valid_text_seq_X_transformed_100.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_valid:
                    ok1=True
                    if a[4:] in unique_list_valid:
                        valid_text_seq_X_transformed_100.iloc[i,j]=a[:4]
                        valid_text_seq_X_transformed_100.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_valid:
                ok1=True
                if a[3:] in unique_list_valid:
                    valid_text_seq_X_transformed_100.iloc[i,j]=a[:3]
                    valid_text_seq_X_transformed_100.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_valid:
                        valid_text_seq_X_transformed_100.iloc[i,j]=a[:5]
                        valid_text_seq_X_transformed_100.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_valid:
                        valid_text_seq_X_transformed_100.iloc[i,j]=a[:4]
                        valid_text_seq_X_transformed_100.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_valid:
                    valid_text_seq_X_transformed_100.iloc[i,j]=a[:3]
                    valid_text_seq_X_transformed_100.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_valid):
    for j in range (0,13):
        a=valid_text_seq_X_transformed_100.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_valid:
                    ok1=True
                    if a[5:] in unique_list_valid:
                        valid_text_seq_X_transformed_100.iloc[i,j]=a[:5]
                        valid_text_seq_X_transformed_100.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_valid:
                    ok1=True
                    if a[4:] in unique_list_valid:
                        valid_text_seq_X_transformed_100.iloc[i,j]=a[:4]
                        valid_text_seq_X_transformed_100.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_valid:
                ok1=True
                if a[3:] in unique_list_valid:
                    valid_text_seq_X_transformed_100.iloc[i,j]=a[:3]
                    valid_text_seq_X_transformed_100.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_valid:
                        valid_text_seq_X_transformed_100.iloc[i,j]=a[:5]
                        valid_text_seq_X_transformed_100.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_valid:
                        valid_text_seq_X_transformed_100.iloc[i,j]=a[:4]
                        valid_text_seq_X_transformed_100.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_valid:
                    valid_text_seq_X_transformed_100.iloc[i,j]=a[:3]
                    valid_text_seq_X_transformed_100.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_valid):
    for j in range (0,13):
        a=valid_text_seq_X_transformed_100.iloc[i,j]
        if a=='':
            b=valid_text_seq_X_transformed_100.iloc[i,j-1]
            valid_text_seq_X_transformed_100.iloc[i,j-1]=b[:4]
            valid_text_seq_X_transformed_100.iloc[i,j]=b[4:]
            break


# #### Encoding Dataset

# In[66]:


train_text_seq_X_transformed_100 =train_text_seq_X_transformed_100.apply(lambda row: ' '.join(row.values), axis=1)
train_text_seq_X_transformed_100 = pd.DataFrame(train_text_seq_X_transformed_100, columns=['text'])


# In[67]:


valid_text_seq_X_transformed_100 =valid_text_seq_X_transformed_100.apply(lambda row: ' '.join(row.values), axis=1)
valid_text_seq_X_transformed_100 = pd.DataFrame(valid_text_seq_X_transformed_100, columns=['text'])


# In[68]:


train_df = train_text_seq_X_transformed_100
valid_df = valid_text_seq_X_transformed_100

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
vocab_dict_100 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_100):
    return df['tokens'].apply(lambda x: [vocab_dict_100[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_100)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_100)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_100) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_100 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model_100')
model_1_100.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_100.summary()

train_labels = train_text_seq_Y_100
valid_labels = valid_text_seq_Y_100
model_1_100.fit(train_padded, train_labels, epochs=10, batch_size=2)

embedding_model_100 = Model(inputs=model_1_100.input, outputs=model_1_100.get_layer("embedding_layer").output)

train_embeddings = embedding_model_100.predict(train_padded)
valid_embeddings = embedding_model_100.predict(valid_padded)


# In[69]:


train_text_seq_X_flattened_100 = train_embeddings.reshape(train_embeddings.shape[0], -1)
train_text_seq_X_encoded_100=pd.DataFrame(train_text_seq_X_flattened_100)

valid_text_seq_X_flattened_100 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)
valid_text_seq_X_encoded_100=pd.DataFrame(valid_text_seq_X_flattened_100)


# ### Feature Standardization

# In[70]:


from sklearn.preprocessing import StandardScaler

scaler_100 = StandardScaler()
train_text_seq_X_encoded_100 = scaler_100.fit_transform(train_text_seq_X_encoded_100)
valid_text_seq_X_encoded_100 = scaler_100.transform(valid_text_seq_X_encoded_100)


# ### Model Training

# In[71]:


model_100=SVC(C=1, degree=2, gamma='scale', kernel='rbf')

model_100.fit(train_text_seq_X_encoded_100, train_text_seq_Y_100)

y_pred_train_100 = model_100.predict(train_text_seq_X_encoded_100)
y_pred_valid_100 = model_100.predict(valid_text_seq_X_encoded_100)


# ### Accuracy checking

# In[72]:


accuracy_100 = accuracy_score(valid_text_seq_Y_100, y_pred_valid_100)
conf_matrix = confusion_matrix(valid_text_seq_Y_100, y_pred_valid_100)

print(f"Validation Accuracy: {accuracy_100*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

# Step 7: Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 80% of Training Data

# ### Transforming Dataset set to a pattern similar to Emoticon Dataset

# #### Training Dataset Transformation

# In[73]:


train_text_seq_X_80 = train_text_seq_X_80.str.replace('15436', 'a')

train_text_seq_X_80=train_text_seq_X_80.to_numpy()
train_text_seq_X_80 = [[sample[:-17], sample[-17:]] for sample in train_text_seq_X_80]
train_text_seq_X_80 = pd.DataFrame(train_text_seq_X_80)

train_text_seq_X_80[1] = train_text_seq_X_80[1].str.replace('1596', 'd')
train_text_seq_X_80[1] = train_text_seq_X_80[1].str.replace('284', 'g')
train_text_seq_X_80[1] = train_text_seq_X_80[1].str.replace('614','e')

for i in range (0,size_train_80):
    if train_text_seq_X_80.iloc[i,1].count('262') != 1:
        n = len(train_text_seq_X_80.iloc[i,1])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_80.iloc[i,1][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2:g[j+2]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_80.iloc[i,1][j:j+3]=='262'):
                a=train_text_seq_X_80.iloc[i,1][:j]
                b=train_text_seq_X_80.iloc[i,1][j:j+3]
                c=train_text_seq_X_80.iloc[i,1][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_80.iloc[i,1]=a+b+c
                break
    else:
        train_text_seq_X_80.iloc[i,1] = train_text_seq_X_80.iloc[i,1].replace('262','f')

train_text_seq_X_80['input_str']=train_text_seq_X_80[0]+train_text_seq_X_80[1]
train_text_seq_X_80=train_text_seq_X_80.drop(columns=[0,1])

train_text_seq_X_80['input_str'] = train_text_seq_X_80['input_str'].str.replace('1596','d')
train_text_seq_X_80['input_str'] = train_text_seq_X_80['input_str'].str.lstrip('0')

for i in range (0,size_train_80):
    if train_text_seq_X_80.iloc[i,0].count('262') == 1 and train_text_seq_X_80.iloc[i,0].count('26262') == 0:
        train_text_seq_X_80.iloc[i,0] = train_text_seq_X_80.iloc[i,0].replace('262','f')
    if train_text_seq_X_80.iloc[i,0].count('464') == 1:
        train_text_seq_X_80.iloc[i,0] = train_text_seq_X_80.iloc[i,0].replace('464','b')
    if train_text_seq_X_80.iloc[i,0].count('422') == 1:
        train_text_seq_X_80.iloc[i,0] = train_text_seq_X_80.iloc[i,0].replace('422','c')
    if train_text_seq_X_80.iloc[i,0].count('614') == 1:
        train_text_seq_X_80.iloc[i,0] = train_text_seq_X_80.iloc[i,0].replace('614','e')

for i in range (0,size_train_80):
    if train_text_seq_X_80.iloc[i,0].count('262') == 1 and train_text_seq_X_80.iloc[i,0].count('26262') == 0 and train_text_seq_X_80.iloc[i,0].count('f')==1:
        train_text_seq_X_80.iloc[i,0] = train_text_seq_X_80.iloc[i,0].replace('262','f')
    if train_text_seq_X_80.iloc[i,0].count('464') == 1 and train_text_seq_X_80.iloc[i,0].count('b')==0:
        train_text_seq_X_80.iloc[i,0] = train_text_seq_X_80.iloc[i,0].replace('464','b')
    if train_text_seq_X_80.iloc[i,0].count('422') == 1 and train_text_seq_X_80.iloc[i,0].count('c')==0:
        train_text_seq_X_80.iloc[i,0] = train_text_seq_X_80.iloc[i,0].replace('422','c')
    if train_text_seq_X_80.iloc[i,0].count('614') == 1 and train_text_seq_X_80.iloc[i,0].count('e')==1:
        train_text_seq_X_80.iloc[i,0] = train_text_seq_X_80.iloc[i,0].replace('614','e')

for i in range (0,size_train_80):
    if train_text_seq_X_80.iloc[i,0].count('b') == 0:
        n = len(train_text_seq_X_80.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_80.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_80.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_80.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_80.iloc[i,0][j:j+3]=='464'):
                a=train_text_seq_X_80.iloc[i,0][:j]
                b=train_text_seq_X_80.iloc[i,0][j:j+3]
                c=train_text_seq_X_80.iloc[i,0][j+3:]
                b=b.replace('464','b')
                train_text_seq_X_80.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_80):
    if train_text_seq_X_80.iloc[i,0].count('c') == 0:
        n = len(train_text_seq_X_80.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_80.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_80.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_80.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_80.iloc[i,0][j:j+3]=='422'):
                a=train_text_seq_X_80.iloc[i,0][:j]
                b=train_text_seq_X_80.iloc[i,0][j:j+3]
                c=train_text_seq_X_80.iloc[i,0][j+3:]
                b=b.replace('422','c')
                train_text_seq_X_80.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_80):
    if train_text_seq_X_80.iloc[i,0].count('e') == 1:
        n = len(train_text_seq_X_80.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_80.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_80.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_80.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_80.iloc[i,0][j:j+3]=='614'):
                a=train_text_seq_X_80.iloc[i,0][:j]
                b=train_text_seq_X_80.iloc[i,0][j:j+3]
                c=train_text_seq_X_80.iloc[i,0][j+3:]
                b=b.replace('614','e')
                train_text_seq_X_80.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_80):
    if train_text_seq_X_80.iloc[i,0].count('f') == 1:
        n = len(train_text_seq_X_80.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_80.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_80.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_80.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_80.iloc[i,0][j:j+3]=='262'):
                a=train_text_seq_X_80.iloc[i,0][:j]
                b=train_text_seq_X_80.iloc[i,0][j:j+3]
                c=train_text_seq_X_80.iloc[i,0][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_80.iloc[i,0]=a+b+c
                break

for i in range(0,size_train_80):
    a=train_text_seq_X_80.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_80.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

for i in range(0,size_train_80):
    k=0
    check=False
    check1=False
    a=train_text_seq_X_80.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0:
                if l<3:
                    check=True
                    break
                else:
                    if l>5:
                        k+=2
                    else:
                        k+=1
                if k>3:
                    check1=True
                    break
                l=0
    if check:
        continue
    if l!=0:
        if l<3:
            check=True
        else:
            if l>5:
                k+=2
            else:
                k+=1
        if k>3:
            check1=True
    if check:
        continue
    if check1:
        if a.count('262')>0:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('262','temp')
            x=x.replace('f','262')
            x=x.replace('temp','f')
            a=x+y
            train_text_seq_X_80.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('422','temp')
            x=x.replace('c','422')
            x=x.replace('temp','c')
            a=x+y
            train_text_seq_X_80.iloc[i,0]=a

for i in range(0,size_train_80):
    check=False
    a=train_text_seq_X_80.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0 and l<3:
                check=True
            l=0
    if a[n-1].isdigit() and l!=0 and l<3:
        check=True
    if check:
        if a.count('262')==0:
            if a.count('614')>0:
                x=a[:-8]
                y=a[-8:]
                x=x.replace('614','nis')
                x=x.replace('e','614')
                x=x.replace('nis','e')
                a=x+y
            elif a.count('f62')>0:
                a=a.replace('f62','26f')
            else:
                a=a.replace('b64','46b')
            train_text_seq_X_80.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            if y.count('262')>0:
                y=y.replace('262','nis')
                y=y.replace('f','262')
                y=y.replace('nis','f')
            if x.count('262')>0:
                x=x.replace('262','nis')
                x=x.replace('f','262')
                x=x.replace('nis','f')
            a=x+y
            train_text_seq_X_80.iloc[i,0]=a

for i in range(0,size_train_80):
    a=train_text_seq_X_80.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_80.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

train_text_seq_X_transformed_80 = pd.DataFrame([[''] * 13] * size_train_80)
value_list_train = []

for i in range (0,size_train_80):
    a=train_text_seq_X_80.iloc[i,0]
    n=len(a)
    k=''
    r=0
    for j in range (0,n):
        if a[j].isdigit():
            k=k+a[j]
        else:
            if len(k)>5:
                if len(k)>10:
                    train_text_seq_X_transformed_80.iloc[i,r]=k
                    r+=3
                else:
                    train_text_seq_X_transformed_80.iloc[i,r]=k
                    r+=2
                train_text_seq_X_transformed_80.iloc[i,r]=a[j]
                r+=1
            else:
                if len(k)==0:
                    train_text_seq_X_transformed_80.iloc[i,r]=a[j]
                    r+=1
                else:
                    value_list_train.append(k)
                    train_text_seq_X_transformed_80.iloc[i,r]=k
                    r+=1
                    train_text_seq_X_transformed_80.iloc[i,r]=a[j]
                    r+=1
            k=''
    if r<13:
        value_list_train.append(k)
        train_text_seq_X_transformed_80.iloc[i,12]=k
unique_list_train=set(value_list_train)

for i in range (0,size_train_80):
    for j in range (0,13):
        a=train_text_seq_X_transformed_80.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_80.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_80.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_80.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_80.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_80.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_80.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_80.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_80.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_80.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_80.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_80.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_80.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_80):
    for j in range (0,13):
        a=train_text_seq_X_transformed_80.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_80.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_80.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_80.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_80.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_80.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_80.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_80.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_80.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_80.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_80.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_80.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_80.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_80):
    for j in range (0,13):
        a=train_text_seq_X_transformed_80.iloc[i,j]
        if a=='':
            b=train_text_seq_X_transformed_80.iloc[i,j-1]
            train_text_seq_X_transformed_80.iloc[i,j-1]=b[:4]
            train_text_seq_X_transformed_80.iloc[i,j]=b[4:]
            break


# #### Validation Dataset Transformation

# In[74]:


valid_text_seq_X_transformed_80 = valid_text_seq_X_transformed_100


# ### Encoding Dataset

# In[75]:


train_text_seq_X_transformed_80 =train_text_seq_X_transformed_80.apply(lambda row: ' '.join(row.values), axis=1)
train_text_seq_X_transformed_80 = pd.DataFrame(train_text_seq_X_transformed_80, columns=['text'])


# In[76]:


train_df = train_text_seq_X_transformed_80
valid_df = valid_text_seq_X_transformed_80

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
vocab_dict_80 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_80):
    return df['tokens'].apply(lambda x: [vocab_dict_80[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_80)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_80)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_80) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_80 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model_80')
model_1_80.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_80.summary()

train_labels = train_text_seq_Y_80
valid_labels = valid_text_seq_Y_80
model_1_80.fit(train_padded, train_labels, epochs=10, batch_size=2)

embedding_model_80 = Model(inputs=model_1_80.input, outputs=model_1_80.get_layer("embedding_layer").output)

train_embeddings = embedding_model_80.predict(train_padded)
valid_embeddings = embedding_model_80.predict(valid_padded)


# In[77]:


train_text_seq_X_flattened_80 = train_embeddings.reshape(train_embeddings.shape[0], -1)
train_text_seq_X_encoded_80=pd.DataFrame(train_text_seq_X_flattened_80)

valid_text_seq_X_flattened_80 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)
valid_text_seq_X_encoded_80=pd.DataFrame(valid_text_seq_X_flattened_80)


# ### Feature Standardization

# In[78]:


from sklearn.preprocessing import StandardScaler

scaler_80 = StandardScaler()
train_text_seq_X_encoded_80 = scaler_80.fit_transform(train_text_seq_X_encoded_80)
valid_text_seq_X_encoded_80 = scaler_80.transform(valid_text_seq_X_encoded_80)


# ### Model Training

# In[79]:


# model_80=LogisticRegression(penalty='l1',C=0.09, solver='liblinear', random_state=1, class_weight='balanced')
model_80=SVC(C=1, degree=2, gamma='scale', kernel='rbf')

model_80.fit(train_text_seq_X_encoded_80, train_text_seq_Y_80)

y_pred_train_80 = model_80.predict(train_text_seq_X_encoded_80)
y_pred_valid_80 = model_80.predict(valid_text_seq_X_encoded_80)


# ### Accuracy Checking

# In[80]:


accuracy_80 = accuracy_score(valid_text_seq_Y_80, y_pred_valid_80)
conf_matrix = confusion_matrix(valid_text_seq_Y_80, y_pred_valid_80)

print(f"Validation Accuracy: {accuracy_80*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

# Step 7: Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 60% Training Data

# ### Transforming Dataset set to a pattern similar to Emoticon Dataset

# #### Training Dataset Transformation

# In[81]:


train_text_seq_X_60 = train_text_seq_X_60.str.replace('15436', 'a')

train_text_seq_X_60=train_text_seq_X_60.to_numpy()
train_text_seq_X_60 = [[sample[:-17], sample[-17:]] for sample in train_text_seq_X_60]
train_text_seq_X_60 = pd.DataFrame(train_text_seq_X_60)

train_text_seq_X_60[1] = train_text_seq_X_60[1].str.replace('1596', 'd')
train_text_seq_X_60[1] = train_text_seq_X_60[1].str.replace('284', 'g')
train_text_seq_X_60[1] = train_text_seq_X_60[1].str.replace('614','e')

for i in range (0,size_train_60):
    if train_text_seq_X_60.iloc[i,1].count('262') != 1:
        n = len(train_text_seq_X_60.iloc[i,1])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_60.iloc[i,1][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2:g[j+2]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_60.iloc[i,1][j:j+3]=='262'):
                a=train_text_seq_X_60.iloc[i,1][:j]
                b=train_text_seq_X_60.iloc[i,1][j:j+3]
                c=train_text_seq_X_60.iloc[i,1][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_60.iloc[i,1]=a+b+c
                break
    else:
        train_text_seq_X_60.iloc[i,1] = train_text_seq_X_60.iloc[i,1].replace('262','f')

train_text_seq_X_60['input_str']=train_text_seq_X_60[0]+train_text_seq_X_60[1]
train_text_seq_X_60=train_text_seq_X_60.drop(columns=[0,1])

train_text_seq_X_60['input_str'] = train_text_seq_X_60['input_str'].str.replace('1596','d')
train_text_seq_X_60['input_str'] = train_text_seq_X_60['input_str'].str.lstrip('0')

for i in range (0,size_train_60):
    if train_text_seq_X_60.iloc[i,0].count('262') == 1 and train_text_seq_X_60.iloc[i,0].count('26262') == 0:
        train_text_seq_X_60.iloc[i,0] = train_text_seq_X_60.iloc[i,0].replace('262','f')
    if train_text_seq_X_60.iloc[i,0].count('464') == 1:
        train_text_seq_X_60.iloc[i,0] = train_text_seq_X_60.iloc[i,0].replace('464','b')
    if train_text_seq_X_60.iloc[i,0].count('422') == 1:
        train_text_seq_X_60.iloc[i,0] = train_text_seq_X_60.iloc[i,0].replace('422','c')
    if train_text_seq_X_60.iloc[i,0].count('614') == 1:
        train_text_seq_X_60.iloc[i,0] = train_text_seq_X_60.iloc[i,0].replace('614','e')

for i in range (0,size_train_60):
    if train_text_seq_X_60.iloc[i,0].count('262') == 1 and train_text_seq_X_60.iloc[i,0].count('26262') == 0 and train_text_seq_X_60.iloc[i,0].count('f')==1:
        train_text_seq_X_60.iloc[i,0] = train_text_seq_X_60.iloc[i,0].replace('262','f')
    if train_text_seq_X_60.iloc[i,0].count('464') == 1 and train_text_seq_X_60.iloc[i,0].count('b')==0:
        train_text_seq_X_60.iloc[i,0] = train_text_seq_X_60.iloc[i,0].replace('464','b')
    if train_text_seq_X_60.iloc[i,0].count('422') == 1 and train_text_seq_X_60.iloc[i,0].count('c')==0:
        train_text_seq_X_60.iloc[i,0] = train_text_seq_X_60.iloc[i,0].replace('422','c')
    if train_text_seq_X_60.iloc[i,0].count('614') == 1 and train_text_seq_X_60.iloc[i,0].count('e')==1:
        train_text_seq_X_60.iloc[i,0] = train_text_seq_X_60.iloc[i,0].replace('614','e')

for i in range (0,size_train_60):
    if train_text_seq_X_60.iloc[i,0].count('b') == 0:
        n = len(train_text_seq_X_60.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_60.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_60.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_60.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_60.iloc[i,0][j:j+3]=='464'):
                a=train_text_seq_X_60.iloc[i,0][:j]
                b=train_text_seq_X_60.iloc[i,0][j:j+3]
                c=train_text_seq_X_60.iloc[i,0][j+3:]
                b=b.replace('464','b')
                train_text_seq_X_60.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_60):
    if train_text_seq_X_60.iloc[i,0].count('c') == 0:
        n = len(train_text_seq_X_60.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_60.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_60.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_60.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_60.iloc[i,0][j:j+3]=='422'):
                a=train_text_seq_X_60.iloc[i,0][:j]
                b=train_text_seq_X_60.iloc[i,0][j:j+3]
                c=train_text_seq_X_60.iloc[i,0][j+3:]
                b=b.replace('422','c')
                train_text_seq_X_60.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_60):
    if train_text_seq_X_60.iloc[i,0].count('e') == 1:
        n = len(train_text_seq_X_60.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_60.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_60.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_60.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_60.iloc[i,0][j:j+3]=='614'):
                a=train_text_seq_X_60.iloc[i,0][:j]
                b=train_text_seq_X_60.iloc[i,0][j:j+3]
                c=train_text_seq_X_60.iloc[i,0][j+3:]
                b=b.replace('614','e')
                train_text_seq_X_60.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_60):
    if train_text_seq_X_60.iloc[i,0].count('f') == 1:
        n = len(train_text_seq_X_60.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_60.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_60.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_60.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_60.iloc[i,0][j:j+3]=='262'):
                a=train_text_seq_X_60.iloc[i,0][:j]
                b=train_text_seq_X_60.iloc[i,0][j:j+3]
                c=train_text_seq_X_60.iloc[i,0][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_60.iloc[i,0]=a+b+c
                break

for i in range(0,size_train_60):
    a=train_text_seq_X_60.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_60.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

for i in range(0,size_train_60):
    k=0
    check=False
    check1=False
    a=train_text_seq_X_60.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0:
                if l<3:
                    check=True
                    break
                else:
                    if l>5:
                        k+=2
                    else:
                        k+=1
                if k>3:
                    check1=True
                    break
                l=0
    if check:
        continue
    if l!=0:
        if l<3:
            check=True
        else:
            if l>5:
                k+=2
            else:
                k+=1
        if k>3:
            check1=True
    if check:
        continue
    if check1:
        if a.count('262')>0:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('262','temp')
            x=x.replace('f','262')
            x=x.replace('temp','f')
            a=x+y
            train_text_seq_X_60.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('422','temp')
            x=x.replace('c','422')
            x=x.replace('temp','c')
            a=x+y
            train_text_seq_X_60.iloc[i,0]=a

for i in range(0,size_train_60):
    check=False
    a=train_text_seq_X_60.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0 and l<3:
                check=True
            l=0
    if a[n-1].isdigit() and l!=0 and l<3:
        check=True
    if check:
        if a.count('262')==0:
            if a.count('614')>0:
                x=a[:-8]
                y=a[-8:]
                x=x.replace('614','nis')
                x=x.replace('e','614')
                x=x.replace('nis','e')
                a=x+y
            elif a.count('f62')>0:
                a=a.replace('f62','26f')
            else:
                a=a.replace('b64','46b')
            train_text_seq_X_60.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            if y.count('262')>0:
                y=y.replace('262','nis')
                y=y.replace('f','262')
                y=y.replace('nis','f')
            if x.count('262')>0:
                x=x.replace('262','nis')
                x=x.replace('f','262')
                x=x.replace('nis','f')
            a=x+y
            train_text_seq_X_60.iloc[i,0]=a

for i in range(0,size_train_60):
    a=train_text_seq_X_60.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_60.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

train_text_seq_X_transformed_60 = pd.DataFrame([[''] * 13] * size_train_60)
value_list_train = []

for i in range (0,size_train_60):
    a=train_text_seq_X_60.iloc[i,0]
    n=len(a)
    k=''
    r=0
    for j in range (0,n):
        if a[j].isdigit():
            k=k+a[j]
        else:
            if len(k)>5:
                if len(k)>10:
                    train_text_seq_X_transformed_60.iloc[i,r]=k
                    r+=3
                else:
                    train_text_seq_X_transformed_60.iloc[i,r]=k
                    r+=2
                train_text_seq_X_transformed_60.iloc[i,r]=a[j]
                r+=1
            else:
                if len(k)==0:
                    train_text_seq_X_transformed_60.iloc[i,r]=a[j]
                    r+=1
                else:
                    value_list_train.append(k)
                    train_text_seq_X_transformed_60.iloc[i,r]=k
                    r+=1
                    train_text_seq_X_transformed_60.iloc[i,r]=a[j]
                    r+=1
            k=''
    if r<13:
        value_list_train.append(k)
        train_text_seq_X_transformed_60.iloc[i,12]=k
unique_list_train=set(value_list_train)

for i in range (0,size_train_60):
    for j in range (0,13):
        a=train_text_seq_X_transformed_60.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_60.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_60.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_60.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_60.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_60.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_60.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_60.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_60.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_60.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_60.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_60.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_60.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_60):
    for j in range (0,13):
        a=train_text_seq_X_transformed_60.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_60.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_60.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_60.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_60.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_60.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_60.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_60.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_60.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_60.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_60.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_60.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_60.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_60):
    for j in range (0,13):
        a=train_text_seq_X_transformed_60.iloc[i,j]
        if a=='':
            b=train_text_seq_X_transformed_60.iloc[i,j-1]
            train_text_seq_X_transformed_60.iloc[i,j-1]=b[:4]
            train_text_seq_X_transformed_60.iloc[i,j]=b[4:]
            break


# #### Validation Dataset Transformation

# In[82]:


valid_text_seq_X_transformed_60 = valid_text_seq_X_transformed_100


# ### Encoding

# In[83]:


train_text_seq_X_transformed_60 =train_text_seq_X_transformed_60.apply(lambda row: ' '.join(row.values), axis=1)
train_text_seq_X_transformed_60 = pd.DataFrame(train_text_seq_X_transformed_60, columns=['text'])


# In[84]:


train_df = train_text_seq_X_transformed_60
valid_df = valid_text_seq_X_transformed_60

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
vocab_dict_60 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_60):
    return df['tokens'].apply(lambda x: [vocab_dict_60[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_60)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_60)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_60) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_60 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model_60')
model_1_60.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_60.summary()

train_labels = train_text_seq_Y_60
valid_labels = valid_text_seq_Y_60
model_1_60.fit(train_padded, train_labels, epochs=10, batch_size=2)

embedding_model_60 = Model(inputs=model_1_60.input, outputs=model_1_60.get_layer("embedding_layer").output)

train_embeddings = embedding_model_60.predict(train_padded)
valid_embeddings = embedding_model_60.predict(valid_padded)


# In[85]:


train_text_seq_X_flattened_60 = train_embeddings.reshape(train_embeddings.shape[0], -1)
train_text_seq_X_encoded_60=pd.DataFrame(train_text_seq_X_flattened_60)

valid_text_seq_X_flattened_60 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)
valid_text_seq_X_encoded_60=pd.DataFrame(valid_text_seq_X_flattened_60)


# ### Feature Standardization

# In[86]:


from sklearn.preprocessing import StandardScaler

scaler_60 = StandardScaler()
train_text_seq_X_encoded_60 = scaler_60.fit_transform(train_text_seq_X_encoded_60)
valid_text_seq_X_encoded_60 = scaler_60.transform(valid_text_seq_X_encoded_60)


# ### Model Training

# In[87]:


# model_60=LogisticRegression(penalty='l1',C=0.09, solver='liblinear', random_state=1, class_weight='balanced')
model_60=SVC(C=1, degree=2, gamma='scale', kernel='rbf')

model_60.fit(train_text_seq_X_encoded_60, train_text_seq_Y_60)

y_pred_train_60 = model_60.predict(train_text_seq_X_encoded_60)
y_pred_valid_60 = model_60.predict(valid_text_seq_X_encoded_60)


# ### Accuracy Checking

# In[88]:


accuracy_60 = accuracy_score(valid_text_seq_Y_60, y_pred_valid_60)
conf_matrix = confusion_matrix(valid_text_seq_Y_60, y_pred_valid_60)

print(f"Validation Accuracy: {accuracy_60*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

# Step 7: Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 40% Training Data

# ### Transforming Dataset set to a pattern similar to Emoticon Dataset

# #### Training Dataset Transformation

# In[89]:


train_text_seq_X_40 = train_text_seq_X_40.str.replace('15436', 'a')

train_text_seq_X_40=train_text_seq_X_40.to_numpy()
train_text_seq_X_40 = [[sample[:-17], sample[-17:]] for sample in train_text_seq_X_40]
train_text_seq_X_40 = pd.DataFrame(train_text_seq_X_40)

train_text_seq_X_40[1] = train_text_seq_X_40[1].str.replace('1596', 'd')
train_text_seq_X_40[1] = train_text_seq_X_40[1].str.replace('284', 'g')
train_text_seq_X_40[1] = train_text_seq_X_40[1].str.replace('614','e')

for i in range (0,size_train_40):
    if train_text_seq_X_40.iloc[i,1].count('262') != 1:
        n = len(train_text_seq_X_40.iloc[i,1])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_40.iloc[i,1][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2:g[j+2]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_40.iloc[i,1][j:j+3]=='262'):
                a=train_text_seq_X_40.iloc[i,1][:j]
                b=train_text_seq_X_40.iloc[i,1][j:j+3]
                c=train_text_seq_X_40.iloc[i,1][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_40.iloc[i,1]=a+b+c
                break
    else:
        train_text_seq_X_40.iloc[i,1] = train_text_seq_X_40.iloc[i,1].replace('262','f')

train_text_seq_X_40['input_str']=train_text_seq_X_40[0]+train_text_seq_X_40[1]
train_text_seq_X_40=train_text_seq_X_40.drop(columns=[0,1])

train_text_seq_X_40['input_str'] = train_text_seq_X_40['input_str'].str.replace('1596','d')
train_text_seq_X_40['input_str'] = train_text_seq_X_40['input_str'].str.lstrip('0')

for i in range (0,size_train_40):
    if train_text_seq_X_40.iloc[i,0].count('262') == 1 and train_text_seq_X_40.iloc[i,0].count('26262') == 0:
        train_text_seq_X_40.iloc[i,0] = train_text_seq_X_40.iloc[i,0].replace('262','f')
    if train_text_seq_X_40.iloc[i,0].count('464') == 1:
        train_text_seq_X_40.iloc[i,0] = train_text_seq_X_40.iloc[i,0].replace('464','b')
    if train_text_seq_X_40.iloc[i,0].count('422') == 1:
        train_text_seq_X_40.iloc[i,0] = train_text_seq_X_40.iloc[i,0].replace('422','c')
    if train_text_seq_X_40.iloc[i,0].count('614') == 1:
        train_text_seq_X_40.iloc[i,0] = train_text_seq_X_40.iloc[i,0].replace('614','e')

for i in range (0,size_train_40):
    if train_text_seq_X_40.iloc[i,0].count('262') == 1 and train_text_seq_X_40.iloc[i,0].count('26262') == 0 and train_text_seq_X_40.iloc[i,0].count('f')==1:
        train_text_seq_X_40.iloc[i,0] = train_text_seq_X_40.iloc[i,0].replace('262','f')
    if train_text_seq_X_40.iloc[i,0].count('464') == 1 and train_text_seq_X_40.iloc[i,0].count('b')==0:
        train_text_seq_X_40.iloc[i,0] = train_text_seq_X_40.iloc[i,0].replace('464','b')
    if train_text_seq_X_40.iloc[i,0].count('422') == 1 and train_text_seq_X_40.iloc[i,0].count('c')==0:
        train_text_seq_X_40.iloc[i,0] = train_text_seq_X_40.iloc[i,0].replace('422','c')
    if train_text_seq_X_40.iloc[i,0].count('614') == 1 and train_text_seq_X_40.iloc[i,0].count('e')==1:
        train_text_seq_X_40.iloc[i,0] = train_text_seq_X_40.iloc[i,0].replace('614','e')

for i in range (0,size_train_40):
    if train_text_seq_X_40.iloc[i,0].count('b') == 0:
        n = len(train_text_seq_X_40.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_40.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_40.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_40.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_40.iloc[i,0][j:j+3]=='464'):
                a=train_text_seq_X_40.iloc[i,0][:j]
                b=train_text_seq_X_40.iloc[i,0][j:j+3]
                c=train_text_seq_X_40.iloc[i,0][j+3:]
                b=b.replace('464','b')
                train_text_seq_X_40.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_40):
    if train_text_seq_X_40.iloc[i,0].count('c') == 0:
        n = len(train_text_seq_X_40.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_40.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_40.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_40.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_40.iloc[i,0][j:j+3]=='422'):
                a=train_text_seq_X_40.iloc[i,0][:j]
                b=train_text_seq_X_40.iloc[i,0][j:j+3]
                c=train_text_seq_X_40.iloc[i,0][j+3:]
                b=b.replace('422','c')
                train_text_seq_X_40.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_40):
    if train_text_seq_X_40.iloc[i,0].count('e') == 1:
        n = len(train_text_seq_X_40.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_40.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_40.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_40.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_40.iloc[i,0][j:j+3]=='614'):
                a=train_text_seq_X_40.iloc[i,0][:j]
                b=train_text_seq_X_40.iloc[i,0][j:j+3]
                c=train_text_seq_X_40.iloc[i,0][j+3:]
                b=b.replace('614','e')
                train_text_seq_X_40.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_40):
    if train_text_seq_X_40.iloc[i,0].count('f') == 1:
        n = len(train_text_seq_X_40.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_40.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_40.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_40.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_40.iloc[i,0][j:j+3]=='262'):
                a=train_text_seq_X_40.iloc[i,0][:j]
                b=train_text_seq_X_40.iloc[i,0][j:j+3]
                c=train_text_seq_X_40.iloc[i,0][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_40.iloc[i,0]=a+b+c
                break

for i in range(0,size_train_40):
    a=train_text_seq_X_40.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_40.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

for i in range(0,size_train_40):
    k=0
    check=False
    check1=False
    a=train_text_seq_X_40.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0:
                if l<3:
                    check=True
                    break
                else:
                    if l>5:
                        k+=2
                    else:
                        k+=1
                if k>3:
                    check1=True
                    break
                l=0
    if check:
        continue
    if l!=0:
        if l<3:
            check=True
        else:
            if l>5:
                k+=2
            else:
                k+=1
        if k>3:
            check1=True
    if check:
        continue
    if check1:
        if a.count('262')>0:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('262','temp')
            x=x.replace('f','262')
            x=x.replace('temp','f')
            a=x+y
            train_text_seq_X_40.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('422','temp')
            x=x.replace('c','422')
            x=x.replace('temp','c')
            a=x+y
            train_text_seq_X_40.iloc[i,0]=a

for i in range(0,size_train_40):
    check=False
    a=train_text_seq_X_40.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0 and l<3:
                check=True
            l=0
    if a[n-1].isdigit() and l!=0 and l<3:
        check=True
    if check:
        if a.count('262')==0:
            if a.count('614')>0:
                x=a[:-8]
                y=a[-8:]
                x=x.replace('614','nis')
                x=x.replace('e','614')
                x=x.replace('nis','e')
                a=x+y
            elif a.count('f62')>0:
                a=a.replace('f62','26f')
            else:
                a=a.replace('b64','46b')
            train_text_seq_X_40.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            if y.count('262')>0:
                y=y.replace('262','nis')
                y=y.replace('f','262')
                y=y.replace('nis','f')
            if x.count('262')>0:
                x=x.replace('262','nis')
                x=x.replace('f','262')
                x=x.replace('nis','f')
            a=x+y
            train_text_seq_X_40.iloc[i,0]=a

for i in range(0,size_train_40):
    a=train_text_seq_X_40.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_40.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

train_text_seq_X_transformed_40 = pd.DataFrame([[''] * 13] * size_train_40)
value_list_train = []

for i in range (0,size_train_40):
    a=train_text_seq_X_40.iloc[i,0]
    n=len(a)
    k=''
    r=0
    for j in range (0,n):
        if a[j].isdigit():
            k=k+a[j]
        else:
            if len(k)>5:
                if len(k)>10:
                    train_text_seq_X_transformed_40.iloc[i,r]=k
                    r+=3
                else:
                    train_text_seq_X_transformed_40.iloc[i,r]=k
                    r+=2
                train_text_seq_X_transformed_40.iloc[i,r]=a[j]
                r+=1
            else:
                if len(k)==0:
                    train_text_seq_X_transformed_40.iloc[i,r]=a[j]
                    r+=1
                else:
                    value_list_train.append(k)
                    train_text_seq_X_transformed_40.iloc[i,r]=k
                    r+=1
                    train_text_seq_X_transformed_40.iloc[i,r]=a[j]
                    r+=1
            k=''
    if r<13:
        value_list_train.append(k)
        train_text_seq_X_transformed_40.iloc[i,12]=k
unique_list_train=set(value_list_train)

for i in range (0,size_train_40):
    for j in range (0,13):
        a=train_text_seq_X_transformed_40.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_40.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_40.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_40.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_40.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_40.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_40.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_40.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_40.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_40.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_40.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_40.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_40.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_40):
    for j in range (0,13):
        a=train_text_seq_X_transformed_40.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_40.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_40.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_40.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_40.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_40.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_40.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_40.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_40.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_40.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_40.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_40.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_40.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_40):
    for j in range (0,13):
        a=train_text_seq_X_transformed_40.iloc[i,j]
        if a=='':
            b=train_text_seq_X_transformed_40.iloc[i,j-1]
            train_text_seq_X_transformed_40.iloc[i,j-1]=b[:4]
            train_text_seq_X_transformed_40.iloc[i,j]=b[4:]
            break


# #### Validation Dataset Transformation

# In[90]:


valid_text_seq_X_transformed_40 = valid_text_seq_X_transformed_100


# ### Encoding

# In[91]:


train_text_seq_X_transformed_40 =train_text_seq_X_transformed_40.apply(lambda row: ' '.join(row.values), axis=1)
train_text_seq_X_transformed_40 = pd.DataFrame(train_text_seq_X_transformed_40, columns=['text'])


# In[92]:


train_df = train_text_seq_X_transformed_40
valid_df = valid_text_seq_X_transformed_40

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
vocab_dict_40 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_40):
    return df['tokens'].apply(lambda x: [vocab_dict_40[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_40)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_40)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_40) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_40 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model_40')
model_1_40.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_40.summary()

train_labels = train_text_seq_Y_40
valid_labels = valid_text_seq_Y_40
model_1_40.fit(train_padded, train_labels, epochs=10, batch_size=2)

embedding_model_40 = Model(inputs=model_1_40.input, outputs=model_1_40.get_layer("embedding_layer").output)

train_embeddings = embedding_model_40.predict(train_padded)
valid_embeddings = embedding_model_40.predict(valid_padded)


# In[93]:


train_text_seq_X_flattened_40 = train_embeddings.reshape(train_embeddings.shape[0], -1)
train_text_seq_X_encoded_40=pd.DataFrame(train_text_seq_X_flattened_40)

valid_text_seq_X_flattened_40 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)
valid_text_seq_X_encoded_40=pd.DataFrame(valid_text_seq_X_flattened_40)


# ### Feature Standardization

# In[94]:


from sklearn.preprocessing import StandardScaler

scaler_40 = StandardScaler()
train_text_seq_X_encoded_40 = scaler_40.fit_transform(train_text_seq_X_encoded_40)
valid_text_seq_X_encoded_40 = scaler_40.transform(valid_text_seq_X_encoded_40)


# ### Model Training

# In[95]:


model_40=SVC(C=1, degree=2, gamma='scale', kernel='rbf')

model_40.fit(train_text_seq_X_encoded_40, train_text_seq_Y_40)

y_pred_train_40 = model_40.predict(train_text_seq_X_encoded_40)
y_pred_valid_40 = model_40.predict(valid_text_seq_X_encoded_40)


# ### Accuracy Checking

# In[96]:


accuracy_40 = accuracy_score(valid_text_seq_Y_40, y_pred_valid_40)
conf_matrix = confusion_matrix(valid_text_seq_Y_40, y_pred_valid_40)

print(f"Validation Accuracy: {accuracy_40*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

# Step 7: Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 20% Training Data

# ### Transforming Dataset set to a pattern similar to Emoticon Dataset

# #### Training Dataset Transformation

# In[97]:


train_text_seq_X_20 = train_text_seq_X_20.str.replace('15436', 'a')

train_text_seq_X_20=train_text_seq_X_20.to_numpy()
train_text_seq_X_20 = [[sample[:-17], sample[-17:]] for sample in train_text_seq_X_20]
train_text_seq_X_20 = pd.DataFrame(train_text_seq_X_20)

train_text_seq_X_20[1] = train_text_seq_X_20[1].str.replace('1596', 'd')
train_text_seq_X_20[1] = train_text_seq_X_20[1].str.replace('284', 'g')
train_text_seq_X_20[1] = train_text_seq_X_20[1].str.replace('614','e')

for i in range (0,size_train_20):
    if train_text_seq_X_20.iloc[i,1].count('262') != 1:
        n = len(train_text_seq_X_20.iloc[i,1])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_20.iloc[i,1][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2:g[j+2]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_20.iloc[i,1][j:j+3]=='262'):
                a=train_text_seq_X_20.iloc[i,1][:j]
                b=train_text_seq_X_20.iloc[i,1][j:j+3]
                c=train_text_seq_X_20.iloc[i,1][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_20.iloc[i,1]=a+b+c
                break
    else:
        train_text_seq_X_20.iloc[i,1] = train_text_seq_X_20.iloc[i,1].replace('262','f')

train_text_seq_X_20['input_str']=train_text_seq_X_20[0]+train_text_seq_X_20[1]
train_text_seq_X_20=train_text_seq_X_20.drop(columns=[0,1])

train_text_seq_X_20['input_str'] = train_text_seq_X_20['input_str'].str.replace('1596','d')
train_text_seq_X_20['input_str'] = train_text_seq_X_20['input_str'].str.lstrip('0')

for i in range (0,size_train_20):
    if train_text_seq_X_20.iloc[i,0].count('262') == 1 and train_text_seq_X_20.iloc[i,0].count('26262') == 0:
        train_text_seq_X_20.iloc[i,0] = train_text_seq_X_20.iloc[i,0].replace('262','f')
    if train_text_seq_X_20.iloc[i,0].count('464') == 1:
        train_text_seq_X_20.iloc[i,0] = train_text_seq_X_20.iloc[i,0].replace('464','b')
    if train_text_seq_X_20.iloc[i,0].count('422') == 1:
        train_text_seq_X_20.iloc[i,0] = train_text_seq_X_20.iloc[i,0].replace('422','c')
    if train_text_seq_X_20.iloc[i,0].count('614') == 1:
        train_text_seq_X_20.iloc[i,0] = train_text_seq_X_20.iloc[i,0].replace('614','e')

for i in range (0,size_train_20):
    if train_text_seq_X_20.iloc[i,0].count('262') == 1 and train_text_seq_X_20.iloc[i,0].count('26262') == 0 and train_text_seq_X_20.iloc[i,0].count('f')==1:
        train_text_seq_X_20.iloc[i,0] = train_text_seq_X_20.iloc[i,0].replace('262','f')
    if train_text_seq_X_20.iloc[i,0].count('464') == 1 and train_text_seq_X_20.iloc[i,0].count('b')==0:
        train_text_seq_X_20.iloc[i,0] = train_text_seq_X_20.iloc[i,0].replace('464','b')
    if train_text_seq_X_20.iloc[i,0].count('422') == 1 and train_text_seq_X_20.iloc[i,0].count('c')==0:
        train_text_seq_X_20.iloc[i,0] = train_text_seq_X_20.iloc[i,0].replace('422','c')
    if train_text_seq_X_20.iloc[i,0].count('614') == 1 and train_text_seq_X_20.iloc[i,0].count('e')==1:
        train_text_seq_X_20.iloc[i,0] = train_text_seq_X_20.iloc[i,0].replace('614','e')

for i in range (0,size_train_20):
    if train_text_seq_X_20.iloc[i,0].count('b') == 0:
        n = len(train_text_seq_X_20.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_20.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_20.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_20.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_20.iloc[i,0][j:j+3]=='464'):
                a=train_text_seq_X_20.iloc[i,0][:j]
                b=train_text_seq_X_20.iloc[i,0][j:j+3]
                c=train_text_seq_X_20.iloc[i,0][j+3:]
                b=b.replace('464','b')
                train_text_seq_X_20.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_20):
    if train_text_seq_X_20.iloc[i,0].count('c') == 0:
        n = len(train_text_seq_X_20.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_20.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_20.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_20.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_20.iloc[i,0][j:j+3]=='422'):
                a=train_text_seq_X_20.iloc[i,0][:j]
                b=train_text_seq_X_20.iloc[i,0][j:j+3]
                c=train_text_seq_X_20.iloc[i,0][j+3:]
                b=b.replace('422','c')
                train_text_seq_X_20.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_20):
    if train_text_seq_X_20.iloc[i,0].count('e') == 1:
        n = len(train_text_seq_X_20.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_20.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_20.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_20.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_20.iloc[i,0][j:j+3]=='614'):
                a=train_text_seq_X_20.iloc[i,0][:j]
                b=train_text_seq_X_20.iloc[i,0][j:j+3]
                c=train_text_seq_X_20.iloc[i,0][j+3:]
                b=b.replace('614','e')
                train_text_seq_X_20.iloc[i,0]=a+b+c
                break

for i in range (0,size_train_20):
    if train_text_seq_X_20.iloc[i,0].count('f') == 1:
        n = len(train_text_seq_X_20.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not train_text_seq_X_20.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and train_text_seq_X_20.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and train_text_seq_X_20.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (train_text_seq_X_20.iloc[i,0][j:j+3]=='262'):
                a=train_text_seq_X_20.iloc[i,0][:j]
                b=train_text_seq_X_20.iloc[i,0][j:j+3]
                c=train_text_seq_X_20.iloc[i,0][j+3:]
                b=b.replace('262','f')
                train_text_seq_X_20.iloc[i,0]=a+b+c
                break

for i in range(0,size_train_20):
    a=train_text_seq_X_20.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_20.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

for i in range(0,size_train_20):
    k=0
    check=False
    check1=False
    a=train_text_seq_X_20.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0:
                if l<3:
                    check=True
                    break
                else:
                    if l>5:
                        k+=2
                    else:
                        k+=1
                if k>3:
                    check1=True
                    break
                l=0
    if check:
        continue
    if l!=0:
        if l<3:
            check=True
        else:
            if l>5:
                k+=2
            else:
                k+=1
        if k>3:
            check1=True
    if check:
        continue
    if check1:
        if a.count('262')>0:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('262','temp')
            x=x.replace('f','262')
            x=x.replace('temp','f')
            a=x+y
            train_text_seq_X_20.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('422','temp')
            x=x.replace('c','422')
            x=x.replace('temp','c')
            a=x+y
            train_text_seq_X_20.iloc[i,0]=a

for i in range(0,size_train_20):
    check=False
    a=train_text_seq_X_20.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0 and l<3:
                check=True
            l=0
    if a[n-1].isdigit() and l!=0 and l<3:
        check=True
    if check:
        if a.count('262')==0:
            if a.count('614')>0:
                x=a[:-8]
                y=a[-8:]
                x=x.replace('614','nis')
                x=x.replace('e','614')
                x=x.replace('nis','e')
                a=x+y
            elif a.count('f62')>0:
                a=a.replace('f62','26f')
            else:
                a=a.replace('b64','46b')
            train_text_seq_X_20.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            if y.count('262')>0:
                y=y.replace('262','nis')
                y=y.replace('f','262')
                y=y.replace('nis','f')
            if x.count('262')>0:
                x=x.replace('262','nis')
                x=x.replace('f','262')
                x=x.replace('nis','f')
            a=x+y
            train_text_seq_X_20.iloc[i,0]=a

for i in range(0,size_train_20):
    a=train_text_seq_X_20.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                train_text_seq_X_20.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

train_text_seq_X_transformed_20 = pd.DataFrame([[''] * 13] * size_train_20)
value_list_train = []

for i in range (0,size_train_20):
    a=train_text_seq_X_20.iloc[i,0]
    n=len(a)
    k=''
    r=0
    for j in range (0,n):
        if a[j].isdigit():
            k=k+a[j]
        else:
            if len(k)>5:
                if len(k)>10:
                    train_text_seq_X_transformed_20.iloc[i,r]=k
                    r+=3
                else:
                    train_text_seq_X_transformed_20.iloc[i,r]=k
                    r+=2
                train_text_seq_X_transformed_20.iloc[i,r]=a[j]
                r+=1
            else:
                if len(k)==0:
                    train_text_seq_X_transformed_20.iloc[i,r]=a[j]
                    r+=1
                else:
                    value_list_train.append(k)
                    train_text_seq_X_transformed_20.iloc[i,r]=k
                    r+=1
                    train_text_seq_X_transformed_20.iloc[i,r]=a[j]
                    r+=1
            k=''
    if r<13:
        value_list_train.append(k)
        train_text_seq_X_transformed_20.iloc[i,12]=k
unique_list_train=set(value_list_train)

for i in range (0,size_train_20):
    for j in range (0,13):
        a=train_text_seq_X_transformed_20.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_20.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_20.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_20.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_20.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_20.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_20.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_20.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_20.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_20.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_20.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_20.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_20.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_20):
    for j in range (0,13):
        a=train_text_seq_X_transformed_20.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_train:
                    ok1=True
                    if a[5:] in unique_list_train:
                        train_text_seq_X_transformed_20.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_20.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_train:
                    ok1=True
                    if a[4:] in unique_list_train:
                        train_text_seq_X_transformed_20.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_20.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_train:
                ok1=True
                if a[3:] in unique_list_train:
                    train_text_seq_X_transformed_20.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_20.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_train:
                        train_text_seq_X_transformed_20.iloc[i,j]=a[:5]
                        train_text_seq_X_transformed_20.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_train:
                        train_text_seq_X_transformed_20.iloc[i,j]=a[:4]
                        train_text_seq_X_transformed_20.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_train:
                    train_text_seq_X_transformed_20.iloc[i,j]=a[:3]
                    train_text_seq_X_transformed_20.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_train_20):
    for j in range (0,13):
        a=train_text_seq_X_transformed_20.iloc[i,j]
        if a=='':
            b=train_text_seq_X_transformed_20.iloc[i,j-1]
            train_text_seq_X_transformed_20.iloc[i,j-1]=b[:4]
            train_text_seq_X_transformed_20.iloc[i,j]=b[4:]
            break


# #### Validation Dataset transformation

# In[98]:


valid_text_seq_X_transformed_20 = valid_text_seq_X_transformed_100


# ### Encoding

# In[99]:


train_text_seq_X_transformed_20 =train_text_seq_X_transformed_20.apply(lambda row: ' '.join(row.values), axis=1)
train_text_seq_X_transformed_20 = pd.DataFrame(train_text_seq_X_transformed_20, columns=['text'])


# In[100]:


train_df = train_text_seq_X_transformed_20
valid_df = valid_text_seq_X_transformed_20

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
vocab_dict_20 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_20):
    return df['tokens'].apply(lambda x: [vocab_dict_20[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_20)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_20)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_20) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_20 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model_20')
model_1_20.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_20.summary()

train_labels = train_text_seq_Y_20
valid_labels = valid_text_seq_Y_20
model_1_20.fit(train_padded, train_labels, epochs=10, batch_size=2)

embedding_model_20 = Model(inputs=model_1_20.input, outputs=model_1_20.get_layer("embedding_layer").output)

train_embeddings = embedding_model_20.predict(train_padded)
valid_embeddings = embedding_model_20.predict(valid_padded)


# In[101]:


train_text_seq_X_flattened_20 = train_embeddings.reshape(train_embeddings.shape[0], -1)
train_text_seq_X_encoded_20=pd.DataFrame(train_text_seq_X_flattened_20)

valid_text_seq_X_flattened_20 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)
valid_text_seq_X_encoded_20=pd.DataFrame(valid_text_seq_X_flattened_20)


# ### Feature Standardization

# In[102]:


from sklearn.preprocessing import StandardScaler

scaler_20 = StandardScaler()
train_text_seq_X_encoded_20 = scaler_20.fit_transform(train_text_seq_X_encoded_20)
valid_text_seq_X_encoded_20 = scaler_20.transform(valid_text_seq_X_encoded_20)


# ### Model Training

# In[103]:


# model_20=LogisticRegression(penalty='l1',C=0.09, solver='liblinear', random_state=1, class_weight='balanced')
model_20=SVC(C=1, degree=2, gamma='scale', kernel='rbf')

model_20.fit(train_text_seq_X_encoded_20, train_text_seq_Y_20)

y_pred_train_20 = model_20.predict(train_text_seq_X_encoded_20)
y_pred_valid_20 = model_20.predict(valid_text_seq_X_encoded_20)


# ### Accuracy Checking

# In[104]:


accuracy_20 = accuracy_score(valid_text_seq_Y_20, y_pred_valid_20)
conf_matrix = confusion_matrix(valid_text_seq_Y_20, y_pred_valid_20)

print(f"Validation Accuracy: {accuracy_20*100:.4f}")
print("Validation Confusion Matrix:")
print(conf_matrix)

# Step 7: Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## Accuracy Variation plot for different % of data

# In[109]:


accuracy_scores = [accuracy_20*100, accuracy_40*100, accuracy_60*100, accuracy_80*100, accuracy_100*100]
percentage_of_data = [20, 40, 60, 80, 100]

plt.plot(percentage_of_data, accuracy_scores, color='red', marker='o')

plt.title('Text Sequence Dataset accuracies across different percentage of Training Data')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy Scores')

# plt.ylim([95, 100])

plt.grid(True)
plt.show()


# ## Test Dataset prediction

# In[106]:


test_text_seq_df = pd.read_csv("datasets/test/test_text_seq.csv")

size_test = len(test_text_seq_df)

test_text_seq_df['input_str'] = test_text_seq_df['input_str'].str.replace('15436', 'a')

test_text_seq_X = test_text_seq_df['input_str']

test_text_seq_X=test_text_seq_X.to_numpy()
test_text_seq_X = [[sample[:-17], sample[-17:]] for sample in test_text_seq_X]
test_text_seq_X = pd.DataFrame(test_text_seq_X)

test_text_seq_X[1] = test_text_seq_X[1].str.replace('1596', 'd')
test_text_seq_X[1] = test_text_seq_X[1].str.replace('284', 'g')
test_text_seq_X[1] = test_text_seq_X[1].str.replace('614','e')

for i in range (0,size_test):
    if test_text_seq_X.iloc[i,1].count('262') != 1:
        n = len(test_text_seq_X.iloc[i,1])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not test_text_seq_X.iloc[i,1][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2:g[j+2]=0
        for j in range (0,n):
            if g[j] and (test_text_seq_X.iloc[i,1][j:j+3]=='262'):
                a=test_text_seq_X.iloc[i,1][:j]
                b=test_text_seq_X.iloc[i,1][j:j+3]
                c=test_text_seq_X.iloc[i,1][j+3:]
                b=b.replace('262','f')
                test_text_seq_X.iloc[i,1]=a+b+c
                break
    else:
        test_text_seq_X.iloc[i,1] = test_text_seq_X.iloc[i,1].replace('262','f')

test_text_seq_X['input_str']=test_text_seq_X[0]+test_text_seq_X[1]
test_text_seq_X=test_text_seq_X.drop(columns=[0,1])

test_text_seq_X['input_str'] = test_text_seq_X['input_str'].str.replace('1596','d')
test_text_seq_X['input_str'] = test_text_seq_X['input_str'].str.lstrip('0')

for i in range (0,size_test):
    if test_text_seq_X.iloc[i,0].count('262') == 1 and test_text_seq_X.iloc[i,0].count('26262') == 0:
        test_text_seq_X.iloc[i,0] = test_text_seq_X.iloc[i,0].replace('262','f')
    if test_text_seq_X.iloc[i,0].count('464') == 1:
        test_text_seq_X.iloc[i,0] = test_text_seq_X.iloc[i,0].replace('464','b')
    if test_text_seq_X.iloc[i,0].count('422') == 1:
        test_text_seq_X.iloc[i,0] = test_text_seq_X.iloc[i,0].replace('422','c')
    if test_text_seq_X.iloc[i,0].count('614') == 1:
        test_text_seq_X.iloc[i,0] = test_text_seq_X.iloc[i,0].replace('614','e')

for i in range (0,size_test):
    if test_text_seq_X.iloc[i,0].count('262') == 1 and test_text_seq_X.iloc[i,0].count('26262') == 0 and test_text_seq_X.iloc[i,0].count('f')==1:
        test_text_seq_X.iloc[i,0] = test_text_seq_X.iloc[i,0].replace('262','f')
    if test_text_seq_X.iloc[i,0].count('464') == 1 and test_text_seq_X.iloc[i,0].count('b')==0:
        test_text_seq_X.iloc[i,0] = test_text_seq_X.iloc[i,0].replace('464','b')
    if test_text_seq_X.iloc[i,0].count('422') == 1 and test_text_seq_X.iloc[i,0].count('c')==0:
        test_text_seq_X.iloc[i,0] = test_text_seq_X.iloc[i,0].replace('422','c')
    if test_text_seq_X.iloc[i,0].count('614') == 1 and test_text_seq_X.iloc[i,0].count('e')==1:
        test_text_seq_X.iloc[i,0] = test_text_seq_X.iloc[i,0].replace('614','e')

for i in range (0,size_test):
    if test_text_seq_X.iloc[i,0].count('b') == 0:
        n = len(test_text_seq_X.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not test_text_seq_X.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and test_text_seq_X.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and test_text_seq_X.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (test_text_seq_X.iloc[i,0][j:j+3]=='464'):
                a=test_text_seq_X.iloc[i,0][:j]
                b=test_text_seq_X.iloc[i,0][j:j+3]
                c=test_text_seq_X.iloc[i,0][j+3:]
                b=b.replace('464','b')
                test_text_seq_X.iloc[i,0]=a+b+c
                break

for i in range (0,size_test):
    if test_text_seq_X.iloc[i,0].count('c') == 0:
        n = len(test_text_seq_X.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not test_text_seq_X.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and test_text_seq_X.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and test_text_seq_X.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (test_text_seq_X.iloc[i,0][j:j+3]=='422'):
                a=test_text_seq_X.iloc[i,0][:j]
                b=test_text_seq_X.iloc[i,0][j:j+3]
                c=test_text_seq_X.iloc[i,0][j+3:]
                b=b.replace('422','c')
                test_text_seq_X.iloc[i,0]=a+b+c
                break

for i in range (0,size_test):
    if test_text_seq_X.iloc[i,0].count('e') == 1:
        n = len(test_text_seq_X.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not test_text_seq_X.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and test_text_seq_X.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and test_text_seq_X.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (test_text_seq_X.iloc[i,0][j:j+3]=='614'):
                a=test_text_seq_X.iloc[i,0][:j]
                b=test_text_seq_X.iloc[i,0][j:j+3]
                c=test_text_seq_X.iloc[i,0][j+3:]
                b=b.replace('614','e')
                test_text_seq_X.iloc[i,0]=a+b+c
                break

for i in range (0,size_test):
    if test_text_seq_X.iloc[i,0].count('f') == 1:
        n = len(test_text_seq_X.iloc[i,0])
        g = [1 for _ in range(n)]
        for j in range(0,n):
            if not test_text_seq_X.iloc[i,0][j].isdigit():
                g[j] = 0
                if j>1:g[j-2]=0
                if j>0:g[j-1]=0
                if j<n-2 and test_text_seq_X.iloc[i,0][j+1].isdigit():g[j+2]=0
                if j<n-3 and test_text_seq_X.iloc[i,0][j+2].isdigit():g[j+3]=0
        for j in range (0,n):
            if g[j] and (test_text_seq_X.iloc[i,0][j:j+3]=='262'):
                a=test_text_seq_X.iloc[i,0][:j]
                b=test_text_seq_X.iloc[i,0][j:j+3]
                c=test_text_seq_X.iloc[i,0][j+3:]
                b=b.replace('262','f')
                test_text_seq_X.iloc[i,0]=a+b+c
                break

for i in range(0,size_test):
    a=test_text_seq_X.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                test_text_seq_X.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

for i in range(0,size_test):
    k=0
    check=False
    check1=False
    a=test_text_seq_X.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0:
                if l<3:
                    check=True
                    break
                else:
                    if l>5:
                        k+=2
                    else:
                        k+=1
                if k>3:
                    check1=True
                    break
                l=0
    if check:
        continue
    if l!=0:
        if l<3:
            check=True
        else:
            if l>5:
                k+=2
            else:
                k+=1
        if k>3:
            check1=True
    if check:
        continue
    if check1:
        if a.count('262')>0:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('262','temp')
            x=x.replace('f','262')
            x=x.replace('temp','f')
            a=x+y
            test_text_seq_X.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            x=x.replace('422','temp')
            x=x.replace('c','422')
            x=x.replace('temp','c')
            a=x+y
            test_text_seq_X.iloc[i,0]=a

for i in range(0,size_test):
    check=False
    a=test_text_seq_X.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            l+=1
        else:
            if l>0 and l<3:
                check=True
            l=0
    if a[n-1].isdigit() and l!=0 and l<3:
        check=True
    if check:
        if a.count('262')==0:
            if a.count('614')>0:
                x=a[:-8]
                y=a[-8:]
                x=x.replace('614','nis')
                x=x.replace('e','614')
                x=x.replace('nis','e')
                a=x+y
            elif a.count('f62')>0:
                a=a.replace('f62','26f')
            else:
                a=a.replace('b64','46b')
            test_text_seq_X.iloc[i,0]=a
        else:
            x=a[:-8]
            y=a[-8:]
            if y.count('262')>0:
                y=y.replace('262','nis')
                y=y.replace('f','262')
                y=y.replace('nis','f')
            if x.count('262')>0:
                x=x.replace('262','nis')
                x=x.replace('f','262')
                x=x.replace('nis','f')
            a=x+y
            test_text_seq_X.iloc[i,0]=a

for i in range(0,size_test):
    a=test_text_seq_X.iloc[i,0]
    l=0
    n=len(a)
    for j in range (0,n):
        if a[j].isdigit():
            if l==0 and a[j]=='0':
                bfr=a[j-1]
                rep=known_substring_2[bfr]
                x=a[:j-1]
                y=rep
                z=a[j:-7].replace(rep,bfr)
                w=a[-7:]
                a=x+y+z+w
                test_text_seq_X.iloc[i,0]=a
                break
            l+=1
        else:
            l=0

test_text_seq_X_transformed = pd.DataFrame([[''] * 13] * size_test)
value_list_test = []

for i in range (0,size_test):
    a=test_text_seq_X.iloc[i,0]
    n=len(a)
    k=''
    r=0
    for j in range (0,n):
        if a[j].isdigit():
            k=k+a[j]
        else:
            if len(k)>5:
                if len(k)>10:
                    test_text_seq_X_transformed.iloc[i,r]=k
                    r+=3
                else:
                    test_text_seq_X_transformed.iloc[i,r]=k
                    r+=2
                test_text_seq_X_transformed.iloc[i,r]=a[j]
                r+=1
            else:
                if len(k)==0:
                    test_text_seq_X_transformed.iloc[i,r]=a[j]
                    r+=1
                else:
                    value_list_test.append(k)
                    test_text_seq_X_transformed.iloc[i,r]=k
                    r+=1
                    test_text_seq_X_transformed.iloc[i,r]=a[j]
                    r+=1
            k=''
    if r<13:
        value_list_test.append(k)
        test_text_seq_X_transformed.iloc[i,12]=k
unique_list_test=set(value_list_test)

for i in range (0,size_test):
    for j in range (0,13):
        a=test_text_seq_X_transformed.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_test:
                    ok1=True
                    if a[5:] in unique_list_test:
                        test_text_seq_X_transformed.iloc[i,j]=a[:5]
                        test_text_seq_X_transformed.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_test:
                    ok1=True
                    if a[4:] in unique_list_test:
                        test_text_seq_X_transformed.iloc[i,j]=a[:4]
                        test_text_seq_X_transformed.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_test:
                ok1=True
                if a[3:] in unique_list_test:
                    test_text_seq_X_transformed.iloc[i,j]=a[:3]
                    test_text_seq_X_transformed.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_test:
                        test_text_seq_X_transformed.iloc[i,j]=a[:5]
                        test_text_seq_X_transformed.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_test:
                        test_text_seq_X_transformed.iloc[i,j]=a[:4]
                        test_text_seq_X_transformed.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_test:
                    test_text_seq_X_transformed.iloc[i,j]=a[:3]
                    test_text_seq_X_transformed.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_test):
    for j in range (0,13):
        a=test_text_seq_X_transformed.iloc[i,j]
        n=len(a)
        if a.isdigit() and n>5:
            ok1=False
            if n>7:
                if a[:5] in unique_list_test:
                    ok1=True
                    if a[5:] in unique_list_test:
                        test_text_seq_X_transformed.iloc[i,j]=a[:5]
                        test_text_seq_X_transformed.iloc[i,j+1]=a[5:]
                        break
            if n>6:
                if a[:4] in unique_list_test:
                    ok1=True
                    if a[4:] in unique_list_test:
                        test_text_seq_X_transformed.iloc[i,j]=a[:4]
                        test_text_seq_X_transformed.iloc[i,j+1]=a[4:]
                        break
            if a[:3] in unique_list_test:
                ok1=True
                if a[3:] in unique_list_test:
                    test_text_seq_X_transformed.iloc[i,j]=a[:3]
                    test_text_seq_X_transformed.iloc[i,j+1]=a[3:]
                    break
            if ok1:
                if n>7:
                    if a[:5] in unique_list_test:
                        test_text_seq_X_transformed.iloc[i,j]=a[:5]
                        test_text_seq_X_transformed.iloc[i,j+1]=a[5:]
                        break
                if n>6:
                    if a[:4] in unique_list_test:
                        test_text_seq_X_transformed.iloc[i,j]=a[:4]
                        test_text_seq_X_transformed.iloc[i,j+1]=a[4:]
                        break
                if a[:3] in unique_list_test:
                    test_text_seq_X_transformed.iloc[i,j]=a[:3]
                    test_text_seq_X_transformed.iloc[i,j+1]=a[3:]
                    break
            break

for i in range (0,size_test):
    for j in range (0,13):
        a=test_text_seq_X_transformed.iloc[i,j]
        if a=='':
            b=test_text_seq_X_transformed.iloc[i,j-1]
            test_text_seq_X_transformed.iloc[i,j-1]=b[:4]
            test_text_seq_X_transformed.iloc[i,j]=b[4:]
            break


# In[107]:


test_text_seq_X_transformed =test_text_seq_X_transformed.apply(lambda row: ' '.join(row.values), axis=1)
test_text_seq_X_transformed = pd.DataFrame(test_text_seq_X_transformed, columns=['text'])


# In[108]:


test_df = test_text_seq_X_transformed

test_df['tokens'] = test_df['text'].apply(lambda x: x.split())

def tokenize_test_data(df, vocab_dict_100):
    return df['tokens'].apply(lambda x: [vocab_dict_100[token] if token in vocab_dict_100 else 0 for token in x])

test_df['tokenized_text'] = tokenize_test_data(test_df, vocab_dict_100)

test_padded = pad_sequences(test_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

test_embeddings = embedding_model_100.predict(test_padded)
print("Test Embeddings Shape:", test_embeddings.shape)

test_X_deep_flattened = test_embeddings.reshape(test_embeddings.shape[0], -1)
test_text_seq_X_encoded=pd.DataFrame(test_X_deep_flattened)

test_text_seq_X_encoded = scaler_100.transform(test_text_seq_X_encoded)

y_pred_test = model_100.predict(test_text_seq_X_encoded)

np.savetxt("pred_textseq.txt", y_pred_test, fmt="%d", delimiter="\n")

#!/usr/bin/env python
# coding: utf-8

# # Combined Dataset

# ## Importing Combined Dataset

# In[174]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (accuracy_score,confusion_matrix,ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler


# In[175]:


train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon']
train_emoticon_Y = train_emoticon_df['label'].to_numpy()

valid_emoticon_df=pd.read_csv("datasets/valid/valid_emoticon.csv")
valid_emoticon_X = valid_emoticon_df['input_emoticon']
valid_emoticon_Y = valid_emoticon_df['label'].to_numpy()


# In[176]:


# Load the dataset
data = np.load('datasets/train/train_feature.npz', allow_pickle=True)
train_deep_X = data['features']
train_deep_Y = data['label']

# Load validation set
valid_data = np.load('datasets/valid/valid_feature.npz', allow_pickle=True)
valid_deep_X = valid_data['features']
valid_deep_Y = valid_data['label']


# ## Taking % of Training Dataset

# In[177]:


train_emoticon_X_100 = train_emoticon_X
train_emoticon_Y_100 = train_emoticon_Y
train_deep_X_100 = train_deep_X
train_emoticon_X_80, train_emoticon_X_20, train_deep_X_80, train_deep_X_20, train_emoticon_Y_80, train_emoticon_Y_20 = train_test_split(train_emoticon_X, train_deep_X, train_emoticon_Y, test_size=0.2, stratify=train_emoticon_Y, random_state=42)
train_emoticon_X_60, train_emoticon_X_40, train_deep_X_60, train_deep_X_40, train_emoticon_Y_60, train_emoticon_Y_40 = train_test_split(train_emoticon_X, train_deep_X, train_emoticon_Y, test_size=0.4, stratify=train_emoticon_Y, random_state=42)


# ## For 100% Training Data

# ### Feature Tansformation and encoding

# #### For emoticon Dataset

# ##### Transformation

# In[178]:


train_emoticon_X_data_100 = [list(input_str) for input_str in train_emoticon_X_100]
train_emoticon_X_data_100 = pd.DataFrame(train_emoticon_X_data_100)
train_emoticon_X_data_100 = train_emoticon_X_data_100.map(ord)
train_emoticon_X_data_100 = train_emoticon_X_data_100.astype(str)
train_emoticon_X_data_100 = train_emoticon_X_data_100.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_100 = pd.DataFrame(train_emoticon_X_data_100, columns=['text'])


# In[179]:


valid_emoticon_X_data_100 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_100 = pd.DataFrame(valid_emoticon_X_data_100)
valid_emoticon_X_data_100 = valid_emoticon_X_data_100.map(ord)
valid_emoticon_X_data_100 = valid_emoticon_X_data_100.astype(str)
valid_emoticon_X_data_100 = valid_emoticon_X_data_100.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_100 = pd.DataFrame(valid_emoticon_X_data_100, columns=['text'])


# ##### Embedding

# In[180]:


train_df = train_emoticon_X_data_100
valid_df = valid_emoticon_X_data_100

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_100 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_100):
    return df['tokens'].apply(lambda x: [vocab_dict_100[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_100)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_100)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_100) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_100 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_100.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_100.summary()

train_labels = train_emoticon_Y_100
valid_labels = valid_emoticon_Y
model_1_100.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_100 = Model(inputs=model_1_100.input, outputs=model_1_100.get_layer("embedding_layer").output)

train_embeddings = embedding_model_100.predict(train_padded)
valid_embeddings = embedding_model_100.predict(valid_padded)


# In[181]:


train_emoticon_X_flattened_100 = train_embeddings.reshape(train_embeddings.shape[0], -1)
valid_emoticon_X_flattened_100 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)

train_emoticon_X_encoded_100=pd.DataFrame(train_emoticon_X_flattened_100)
valid_emoticon_X_encoded_100=pd.DataFrame(valid_emoticon_X_flattened_100)


# ##### Standardization

# In[182]:


scaler_100 = StandardScaler()
train_emoticon_X_encoded_100 = scaler_100.fit_transform(train_emoticon_X_encoded_100)
valid_emoticon_X_encoded_100 = scaler_100.transform(valid_emoticon_X_encoded_100)


# #### For Deep Feature Dataset

# ##### Feature Transformation

# In[183]:


train_X_deep_flattened_100 = train_deep_X_100.reshape(train_deep_X_100.shape[0], -1)
valid_X_deep_flattened_100 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_100.shape)


# ##### Feature Reduction

# In[184]:


pca_100 = PCA(n_components=100)
train_X_deep_flattened_100 = pca_100.fit_transform(train_X_deep_flattened_100)
valid_X_deep_flattened_100 = pca_100.transform(valid_X_deep_flattened_100)


# ### Model Training

# In[185]:


model1_100 = SVC(C=0.1, degree=2, gamma='auto', kernel='rbf', probability=True)
model2_100 = SVC(C=100, degree=2, gamma='auto', kernel='rbf', probability=True)

model1_100.fit(train_emoticon_X_encoded_100, train_emoticon_Y_100)
model2_100.fit(train_X_deep_flattened_100, train_emoticon_Y_100)

P1 = model1_100.predict_proba(train_emoticon_X_encoded_100)[:, 1]
P2 = model2_100.predict_proba(train_X_deep_flattened_100)[:, 1]

meta_X_train = np.column_stack((P1,P2))

meta_model_100 = RandomForestClassifier()
meta_model_100.fit(meta_X_train, train_emoticon_Y_100)


# ### Accuracy Checking

# In[186]:


P1_valid = model1_100.predict_proba(valid_emoticon_X_encoded_100)[:, 1]
P2_valid = model2_100.predict_proba(valid_X_deep_flattened_100)[:, 1]

meta_X_valid = np.column_stack((P1_valid,P2_valid))

y_pred_valid_100 = meta_model_100.predict(meta_X_valid)

accuracy_100 = accuracy_score(valid_emoticon_Y, y_pred_valid_100)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_100)

print(f"Validation Accuracy: {accuracy_100*100:.4f}")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 80% Training Data

# ### Feature Transformation and Encoding

# #### For Emoticon Dataset

# ##### Transformation

# In[187]:


train_emoticon_X_data_80 = [list(input_str) for input_str in train_emoticon_X_80]
train_emoticon_X_data_80 = pd.DataFrame(train_emoticon_X_data_80)
train_emoticon_X_data_80 = train_emoticon_X_data_80.map(ord)
train_emoticon_X_data_80 = train_emoticon_X_data_80.astype(str)
train_emoticon_X_data_80 = train_emoticon_X_data_80.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_80 = pd.DataFrame(train_emoticon_X_data_80, columns=['text'])


# In[188]:


valid_emoticon_X_data_80 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_80 = pd.DataFrame(valid_emoticon_X_data_80)
valid_emoticon_X_data_80 = valid_emoticon_X_data_80.map(ord)
valid_emoticon_X_data_80 = valid_emoticon_X_data_80.astype(str)
valid_emoticon_X_data_80 = valid_emoticon_X_data_80.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_80 = pd.DataFrame(valid_emoticon_X_data_80, columns=['text'])


# ##### Embedding

# In[189]:


train_df = train_emoticon_X_data_80
valid_df = valid_emoticon_X_data_80

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_80 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_80):
    return df['tokens'].apply(lambda x: [vocab_dict_80[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_80)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_80)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_80) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_80 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_80.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_80.summary()

train_labels = train_emoticon_Y_80
valid_labels = valid_emoticon_Y
model_1_80.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_80 = Model(inputs=model_1_80.input, outputs=model_1_80.get_layer("embedding_layer").output)

train_embeddings = embedding_model_80.predict(train_padded)
valid_embeddings = embedding_model_80.predict(valid_padded)


# In[190]:


train_emoticon_X_flattened_80 = train_embeddings.reshape(train_embeddings.shape[0], -1)
valid_emoticon_X_flattened_80 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)

train_emoticon_X_encoded_80=pd.DataFrame(train_emoticon_X_flattened_80)
valid_emoticon_X_encoded_80=pd.DataFrame(valid_emoticon_X_flattened_80)


# ##### Standardization

# In[191]:


scaler_80 = StandardScaler()
train_emoticon_X_encoded_80 = scaler_80.fit_transform(train_emoticon_X_encoded_80)
valid_emoticon_X_encoded_80 = scaler_80.transform(valid_emoticon_X_encoded_80)


# #### For Deep Feature Dataset

# ##### Feature Transformation

# In[192]:


train_X_deep_flattened_80 = train_deep_X_80.reshape(train_deep_X_80.shape[0], -1)
valid_X_deep_flattened_80 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_80.shape)


# ##### Feature Reduction

# In[193]:


pca_80 = PCA(n_components=100)
train_X_deep_flattened_80 = pca_80.fit_transform(train_X_deep_flattened_80)
valid_X_deep_flattened_80 = pca_80.transform(valid_X_deep_flattened_80)


# ### Model Training

# In[194]:


model1_80 = SVC(C=0.1, degree=2, gamma='auto', kernel='rbf', probability=True)
model2_80 = SVC(C=100, degree=2, gamma='auto', kernel='rbf', probability=True)

model1_80.fit(train_emoticon_X_encoded_80, train_emoticon_Y_80)
model2_80.fit(train_X_deep_flattened_80, train_emoticon_Y_80)

P1 = model1_80.predict_proba(train_emoticon_X_encoded_80)[:, 1]
P2 = model2_80.predict_proba(train_X_deep_flattened_80)[:, 1]

meta_X_train = np.column_stack((P1,P2))

meta_model_80 = RandomForestClassifier()
meta_model_80.fit(meta_X_train, train_emoticon_Y_80)


# ### Accuracy Checking

# In[195]:


P1_valid = model1_80.predict_proba(valid_emoticon_X_encoded_80)[:, 1]
P2_valid = model2_80.predict_proba(valid_X_deep_flattened_80)[:, 1]

meta_X_valid = np.column_stack((P1_valid,P2_valid))

y_pred_valid_80 = meta_model_80.predict(meta_X_valid)

accuracy_80 = accuracy_score(valid_emoticon_Y, y_pred_valid_80)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_80)

print(f"Validation Accuracy: {accuracy_80*100:.4f}")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 60% Training Data

# ### Feature Transformation and encoding

# #### For Emoticon Dataset

# ##### Transformation

# In[196]:


train_emoticon_X_data_60 = [list(input_str) for input_str in train_emoticon_X_60]
train_emoticon_X_data_60 = pd.DataFrame(train_emoticon_X_data_60)
train_emoticon_X_data_60 = train_emoticon_X_data_60.map(ord)
train_emoticon_X_data_60 = train_emoticon_X_data_60.astype(str)
train_emoticon_X_data_60 = train_emoticon_X_data_60.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_60 = pd.DataFrame(train_emoticon_X_data_60, columns=['text'])


# In[197]:


valid_emoticon_X_data_60 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_60 = pd.DataFrame(valid_emoticon_X_data_60)
valid_emoticon_X_data_60 = valid_emoticon_X_data_60.map(ord)
valid_emoticon_X_data_60 = valid_emoticon_X_data_60.astype(str)
valid_emoticon_X_data_60 = valid_emoticon_X_data_60.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_60 = pd.DataFrame(valid_emoticon_X_data_60, columns=['text'])


# ##### Embedding

# In[198]:


train_df = train_emoticon_X_data_60
valid_df = valid_emoticon_X_data_60

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_60 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_60):
    return df['tokens'].apply(lambda x: [vocab_dict_60[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_60)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_60)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_60) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_60 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_60.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_60.summary()

train_labels = train_emoticon_Y_60
valid_labels = valid_emoticon_Y
model_1_60.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_60 = Model(inputs=model_1_60.input, outputs=model_1_60.get_layer("embedding_layer").output)

train_embeddings = embedding_model_60.predict(train_padded)
valid_embeddings = embedding_model_60.predict(valid_padded)


# In[199]:


train_emoticon_X_flattened_60 = train_embeddings.reshape(train_embeddings.shape[0], -1)
valid_emoticon_X_flattened_60 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)

train_emoticon_X_encoded_60=pd.DataFrame(train_emoticon_X_flattened_60)
valid_emoticon_X_encoded_60=pd.DataFrame(valid_emoticon_X_flattened_60)


# ##### Standardization

# In[200]:


scaler_60 = StandardScaler()
train_emoticon_X_encoded_60 = scaler_60.fit_transform(train_emoticon_X_encoded_60)
valid_emoticon_X_encoded_60 = scaler_60.transform(valid_emoticon_X_encoded_60)


# #### For Deep Feature Dataset

# ##### Feature Transformation

# In[201]:


train_X_deep_flattened_60 = train_deep_X_60.reshape(train_deep_X_60.shape[0], -1)
valid_X_deep_flattened_60 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_60.shape)


# ##### Feature Reduction

# In[202]:


pca_60 = PCA(n_components=100)
train_X_deep_flattened_60 = pca_60.fit_transform(train_X_deep_flattened_60)
valid_X_deep_flattened_60 = pca_60.transform(valid_X_deep_flattened_60)


# ### Model Training

# In[203]:


model1_60 = SVC(C=0.1, degree=2, gamma='auto', kernel='rbf', probability=True)
model2_60 = SVC(C=100, degree=2, gamma='auto', kernel='rbf', probability=True)

model1_60.fit(train_emoticon_X_encoded_60, train_emoticon_Y_60)
model2_60.fit(train_X_deep_flattened_60, train_emoticon_Y_60)

P1 = model1_60.predict_proba(train_emoticon_X_encoded_60)[:, 1]
P2 = model2_60.predict_proba(train_X_deep_flattened_60)[:, 1]

meta_X_train = np.column_stack((P1,P2))

meta_model_60 = RandomForestClassifier()
meta_model_60.fit(meta_X_train, train_emoticon_Y_60)


# ### Accuracy Checking

# In[204]:


P1_valid = model1_60.predict_proba(valid_emoticon_X_encoded_60)[:, 1]
P2_valid = model2_60.predict_proba(valid_X_deep_flattened_60)[:, 1]

meta_X_valid = np.column_stack((P1_valid,P2_valid))

y_pred_valid_60 = meta_model_60.predict(meta_X_valid)

accuracy_60 = accuracy_score(valid_emoticon_Y, y_pred_valid_60)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_60)

print(f"Validation Accuracy: {accuracy_60*100:.4f}")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 40% of Training Data

# ### Feature Transformation and Encoding

# #### For emoticon dataset

# ##### Transformation

# In[205]:


train_emoticon_X_data_40 = [list(input_str) for input_str in train_emoticon_X_40]
train_emoticon_X_data_40 = pd.DataFrame(train_emoticon_X_data_40)
train_emoticon_X_data_40 = train_emoticon_X_data_40.map(ord)
train_emoticon_X_data_40 = train_emoticon_X_data_40.astype(str)
train_emoticon_X_data_40 = train_emoticon_X_data_40.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_40 = pd.DataFrame(train_emoticon_X_data_40, columns=['text'])


# In[206]:


valid_emoticon_X_data_40 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_40 = pd.DataFrame(valid_emoticon_X_data_40)
valid_emoticon_X_data_40 = valid_emoticon_X_data_40.map(ord)
valid_emoticon_X_data_40 = valid_emoticon_X_data_40.astype(str)
valid_emoticon_X_data_40 = valid_emoticon_X_data_40.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_40 = pd.DataFrame(valid_emoticon_X_data_40, columns=['text'])


# ##### Embedding

# In[207]:


train_df = train_emoticon_X_data_40
valid_df = valid_emoticon_X_data_40

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_40 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_40):
    return df['tokens'].apply(lambda x: [vocab_dict_40[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_40)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_40)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_40) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_40 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_40.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_40.summary()

train_labels = train_emoticon_Y_40
valid_labels = valid_emoticon_Y
model_1_40.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_40 = Model(inputs=model_1_40.input, outputs=model_1_40.get_layer("embedding_layer").output)

train_embeddings = embedding_model_40.predict(train_padded)
valid_embeddings = embedding_model_40.predict(valid_padded)


# In[208]:


train_emoticon_X_flattened_40 = train_embeddings.reshape(train_embeddings.shape[0], -1)
valid_emoticon_X_flattened_40 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)

train_emoticon_X_encoded_40=pd.DataFrame(train_emoticon_X_flattened_40)
valid_emoticon_X_encoded_40=pd.DataFrame(valid_emoticon_X_flattened_40)


# ##### Standardization

# In[209]:


scaler_40 = StandardScaler()
train_emoticon_X_encoded_40 = scaler_40.fit_transform(train_emoticon_X_encoded_40)
valid_emoticon_X_encoded_40 = scaler_40.transform(valid_emoticon_X_encoded_40)


# #### For Deep Feature Dataset

# ##### Feature Transformation

# In[210]:


train_X_deep_flattened_40 = train_deep_X_40.reshape(train_deep_X_40.shape[0], -1)
valid_X_deep_flattened_40 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_40.shape)


# ##### Feature Reduction

# In[211]:


pca_40 = PCA(n_components=100)
train_X_deep_flattened_40 = pca_40.fit_transform(train_X_deep_flattened_40)
valid_X_deep_flattened_40 = pca_40.transform(valid_X_deep_flattened_40)


# ### Model Training

# In[212]:


model1_40 = SVC(C=0.1, degree=2, gamma='auto', kernel='rbf', probability=True)
model2_40 = SVC(C=100, degree=2, gamma='auto', kernel='rbf', probability=True)

model1_40.fit(train_emoticon_X_encoded_40, train_emoticon_Y_40)
model2_40.fit(train_X_deep_flattened_40, train_emoticon_Y_40)

P1 = model1_40.predict_proba(train_emoticon_X_encoded_40)[:, 1]
P2 = model2_40.predict_proba(train_X_deep_flattened_40)[:, 1]

meta_X_train = np.column_stack((P1,P2))

meta_model_40 = RandomForestClassifier()
meta_model_40.fit(meta_X_train, train_emoticon_Y_40)


# ### Accuracy Checking

# In[213]:


P1_valid = model1_40.predict_proba(valid_emoticon_X_encoded_40)[:, 1]
P2_valid = model2_40.predict_proba(valid_X_deep_flattened_40)[:, 1]

meta_X_valid = np.column_stack((P1_valid,P2_valid))

y_pred_valid_40 = meta_model_40.predict(meta_X_valid)

accuracy_40 = accuracy_score(valid_emoticon_Y, y_pred_valid_40)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_40)

print(f"Validation Accuracy: {accuracy_40*100:.4f}")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## For 20% of Training Data

# ### Feature transformation and Encoding

# #### For Emoticon Dataset

# ##### Transformation

# In[214]:


train_emoticon_X_data_20 = [list(input_str) for input_str in train_emoticon_X_20]
train_emoticon_X_data_20 = pd.DataFrame(train_emoticon_X_data_20)
train_emoticon_X_data_20 = train_emoticon_X_data_20.map(ord)
train_emoticon_X_data_20 = train_emoticon_X_data_20.astype(str)
train_emoticon_X_data_20 = train_emoticon_X_data_20.apply(lambda row: ' '.join(row.values), axis=1)
train_emoticon_X_data_20 = pd.DataFrame(train_emoticon_X_data_20, columns=['text'])


# In[215]:


valid_emoticon_X_data_20 = [list(input_str) for input_str in valid_emoticon_X]
valid_emoticon_X_data_20 = pd.DataFrame(valid_emoticon_X_data_20)
valid_emoticon_X_data_20 = valid_emoticon_X_data_20.map(ord)
valid_emoticon_X_data_20 = valid_emoticon_X_data_20.astype(str)
valid_emoticon_X_data_20 = valid_emoticon_X_data_20.apply(lambda row: ' '.join(row.values), axis=1)
valid_emoticon_X_data_20 = pd.DataFrame(valid_emoticon_X_data_20, columns=['text'])


# ##### Embedding

# In[216]:


train_df = train_emoticon_X_data_20
valid_df = valid_emoticon_X_data_20

combined_vocab = set()
train_df['tokens'] = train_df['text'].apply(lambda x: x.split())
combined_vocab.update(train_df['tokens'].explode().unique())
print(combined_vocab)
valid_df['tokens'] = valid_df['text'].apply(lambda x: x.split())
combined_vocab.update(valid_df['tokens'].explode().unique())
print(combined_vocab)
vocab_dict_20 = {word: idx for idx, word in enumerate(combined_vocab, start=1)}

def tokenize_data(df, vocab_dict_20):
    return df['tokens'].apply(lambda x: [vocab_dict_20[token] for token in x])

train_df['tokenized_text'] = tokenize_data(train_df, vocab_dict_20)
valid_df['tokenized_text'] = tokenize_data(valid_df, vocab_dict_20)
max_length = 13
train_padded = pad_sequences(train_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
valid_padded = pad_sequences(valid_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')

embedding_size = 16
vocab_size = len(vocab_dict_20) + 1

input_layer = Input(shape=(max_length,), dtype='int32')
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_length, name='embedding_layer')(input_layer)
lstm_layer = LSTM(16, return_sequences=True)(embedding_layer)
dropout_layer = Dropout(0.2)(lstm_layer)
flatten_layer = Flatten()(dropout_layer)
dense_layer = Dense(8, activation='relu')(flatten_layer)
output_layer = Dense(1, activation='sigmoid')(dense_layer)

model_1_20 = Model(inputs=input_layer, outputs=output_layer, name='small_text_embedding_model')
model_1_20.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1_20.summary()

train_labels = train_emoticon_Y_20
valid_labels = valid_emoticon_Y
model_1_20.fit(train_padded, train_labels, epochs=6, batch_size=2)

embedding_model_20 = Model(inputs=model_1_20.input, outputs=model_1_20.get_layer("embedding_layer").output)

train_embeddings = embedding_model_20.predict(train_padded)
valid_embeddings = embedding_model_20.predict(valid_padded)


# In[217]:


train_emoticon_X_flattened_20 = train_embeddings.reshape(train_embeddings.shape[0], -1)
valid_emoticon_X_flattened_20 = valid_embeddings.reshape(valid_embeddings.shape[0], -1)

train_emoticon_X_encoded_20=pd.DataFrame(train_emoticon_X_flattened_20)
valid_emoticon_X_encoded_20=pd.DataFrame(valid_emoticon_X_flattened_20)


# ##### Standardization

# In[218]:


scaler_20 = StandardScaler()
train_emoticon_X_encoded_20 = scaler_20.fit_transform(train_emoticon_X_encoded_20)
valid_emoticon_X_encoded_20 = scaler_20.transform(valid_emoticon_X_encoded_20)


# #### For Deep Feature Dataset

# ##### Feature transformation

# In[219]:


train_X_deep_flattened_20 = train_deep_X_20.reshape(train_deep_X_20.shape[0], -1)
valid_X_deep_flattened_20 = valid_deep_X.reshape(valid_deep_X.shape[0], -1)

print(train_X_deep_flattened_20.shape)


# ##### Feature Reduction

# In[220]:


pca_20 = PCA(n_components=100)
train_X_deep_flattened_20 = pca_20.fit_transform(train_X_deep_flattened_20)
valid_X_deep_flattened_20 = pca_20.transform(valid_X_deep_flattened_20)


# ### Model Training

# In[221]:


model1_20 = SVC(C=0.1, degree=2, gamma='auto', kernel='rbf', probability=True)
model2_20 = SVC(C=100, degree=2, gamma='auto', kernel='rbf', probability=True)

model1_20.fit(train_emoticon_X_encoded_20, train_emoticon_Y_20)
model2_20.fit(train_X_deep_flattened_20, train_emoticon_Y_20)

P1 = model1_20.predict_proba(train_emoticon_X_encoded_20)[:, 1]
P2 = model2_20.predict_proba(train_X_deep_flattened_20)[:, 1]

meta_X_train = np.column_stack((P1,P2))

meta_model_20 = RandomForestClassifier()
meta_model_20.fit(meta_X_train, train_emoticon_Y_20)


# ### Accuracy Checking

# In[222]:


P1_valid = model1_20.predict_proba(valid_emoticon_X_encoded_20)[:, 1]
P2_valid = model2_20.predict_proba(valid_X_deep_flattened_20)[:, 1]

meta_X_valid = np.column_stack((P1_valid,P2_valid))

y_pred_valid_20 = meta_model_20.predict(meta_X_valid)

accuracy_20 = accuracy_score(valid_emoticon_Y, y_pred_valid_20)
conf_matrix = confusion_matrix(valid_emoticon_Y, y_pred_valid_20)

print(f"Validation Accuracy: {accuracy_20*100:.4f}")
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')


# ## Accuracy Variation plot for different % of data

# In[229]:


accuracy_scores = [accuracy_20*100, accuracy_40*100, accuracy_60*100, accuracy_80*100, accuracy_100*100]
percentage_of_data = [20, 40, 60, 80, 100]

plt.plot(percentage_of_data, accuracy_scores, color='red', marker='o')

plt.title('Combined Dataset accuracies across different percentage of Training Data')
plt.xlabel('Percentage of Training Data')
plt.ylabel('Accuracy Scores')

# plt.ylim([91, 98])

plt.grid(True)
plt.show()


# ## Prediction for Test Data

# In[224]:


test_emoticon_df=pd.read_csv("datasets/test/test_emoticon.csv")
test_emoticon_X = test_emoticon_df['input_emoticon']

test_emoticon_X_data = [list(input_str) for input_str in test_emoticon_X]
test_emoticon_X_data = pd.DataFrame(test_emoticon_X_data)
test_emoticon_X_data = test_emoticon_X_data.map(ord)
test_emoticon_X_data = test_emoticon_X_data.astype(str)
test_emoticon_X_data = test_emoticon_X_data.apply(lambda row: ' '.join(row.values), axis=1)
test_emoticon_X_data = pd.DataFrame(test_emoticon_X_data, columns=['text'])


# In[225]:


test_df = test_emoticon_X_data
test_df['tokens'] = test_df['text'].apply(lambda x: x.split())
def tokenize_test_data(df, vocab_dict_100):
    return df['tokens'].apply(lambda x: [vocab_dict_100[token] if token in vocab_dict_100 else 0 for token in x])
test_df['tokenized_text'] = tokenize_test_data(test_df, vocab_dict_100)
test_padded = pad_sequences(test_df['tokenized_text'].tolist(), maxlen=max_length, padding='post')
test_embeddings = embedding_model_100.predict(test_padded)
print("Test Embeddings Shape:", test_embeddings.shape)
test_emoticon_X_flattened = test_embeddings.reshape(test_embeddings.shape[0], -1)
test_emoticon_X_encoded=pd.DataFrame(test_emoticon_X_flattened)
test_emoticon_X_encoded = scaler_100.transform(test_emoticon_X_encoded)


# In[226]:


test_data = np.load('datasets/test/test_feature.npz', allow_pickle=True)
test_deep_X = test_data['features']

test_X_deep_flattened = test_deep_X.reshape(test_deep_X.shape[0], -1)
test_X_deep_flattened = pca_100.transform(test_X_deep_flattened)


# In[227]:


P1_test = model1_100.predict_proba(test_emoticon_X_encoded)[:, 1]
P2_test = model2_100.predict_proba(test_X_deep_flattened)[:, 1]

meta_X_test = np.column_stack((P1_test,P2_test))

y_pred_test = meta_model_100.predict(meta_X_test)

np.savetxt("pred_combined.txt", y_pred_test, fmt="%d", delimiter="\n")

