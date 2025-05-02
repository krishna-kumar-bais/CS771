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
