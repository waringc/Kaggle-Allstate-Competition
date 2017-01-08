# Kaggle- Allstate Claims Severity
These are the scripts I created for the Kaggle [Allstate Claims Severity](https://www.kaggle.com/c/allstate-claims-severity) competition.  

My model finished 71st out of 3055 teams.

## Data
The goal of the competition was to predict the severity of claims that would be made to Allstate.  The data is publicly available [here](https://www.kaggle.com/c/allstate-claims-severity/data).

The predictions generated for the competition were evaluated using [Mean Absolute Error] (https://en.wikipedia.org/wiki/Mean_absolute_error) (MAE).


## Dependancies
A combination of R/RStudio and Python were used to create the model.  

The following R packages were used:
* data.table
* dplyr
* Matrix
* xgboost
* Metrics
* caret
* RTsne
* fnn

The following Python packages were used:
* numpy
* pandas
* sklearn
* keras
* lightGBM


## Content

Success in this competition was dependent on [ensembling](http://mlwave.com/kaggle-ensembling-guide/) multiple types of models to generate predictions.  Feature engineering had a limited impact on improving prediction accuracy. I found the best results when averaging XGBoost models with a neural network created using Keras.   First level XGboost and Keras models were created with varying random seeds and model parameters.  The out of fold predictions from these first level models where used as features in second level XGBoost and Keras models.  The second level XGboost and Keras models were averaged to produce the final submission.

The files are organized by the various types of models used to generate predictions.   

### XGBoost (gradient boosted decision trees)

  **xgboost.R**- This script generates the test predictions and out-of-fold predictions for the training data.  XGBoost provided the most accurate solo model for predicting severity(~1107 on private leaderboard).  Feature engineering included creating combinations of categorical variables together and encoding the single and combined categorical features into integers.  To un-skew numeric features a boxcox transformation was applied to make the values more normal.  The value of severities in the train data was also quite skewed and was transformed using log(severity + 200) [transformation](https://www.kaggle.com/c/allstate-claims-severity/forums/t/25062/boxcox-loss-200-cannot-beat-log-loss-200).  I ran this script multiple times with different seed and model parameters.  The out-fold-predictions from these XGBoost models were used as features in a stacking XGBoost model or with Keras or LightGBM.    

### Keras (neural network)

  **keras.py**- This a modified version of a script uploaded to Kaggle kernels
  [here](https://www.kaggle.com/danijelk/allstate-claims-severity/keras-starter-with-bagging-lb-1120-596).  The script uses keras to create a neural network to predict the claims severity.  Although, not as accurate as XGBoost the accuracy of this model was generally quite good (~1110 on private leaderboard).  For data processing, categorical features are one hot encoded and numerical features are scaled to make their distribution more normal.  The script generates both test predictions and out-of-fold training predictions.  The first time I ran this script using only the CPU it took 3 days to run!  After getting CUDA working and training using a GPU the time took about 12 hours.  I ran this script multiple times with different seed values and different numbers of neurons in the network.  The out-fold-predictions from these  models were used as features in a stacking Keras model or with XGBoost or LightGBM.    

### LightGBM (gradient boosted decision trees)

**lightGBM.py**- [LightGBM](https://github.com/Microsoft/LightGBM) is a tree based gradient boosting framework by Microsoft.  At the time of the competition the official R or python package was not available so I used a set of [python bindings](https://github.com/ArdalanM/pyLightGBM).  I wasn't able to get better results with lightGBM than with XGBoost and adding lightGBM to the model ensemble didn't offer any improvement to accuracy.  Though this was my first attempt at using lightGBM and others were able to find success.  I will definitely try using lightGBM in future competitions.       

### t-SNE
**tSNE.R**- [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) that reduces high dimensionality data to 2 or 3 dimensions.  I generated 2D t-SNE features for the claims data and than added the t-SNE features as features in XGBoost and keras models.  I found adding the t-SNE features didn't offer much improvement to the model accuracy and they weren't used in the final ensemble.

### kNN
**knn.R**- This script generates k-nearest neighbours predictions for claims severity.  I ran the script with various values of K from 2 to 2048.  The single kNN results were quite poor and adding them to the ensemble didn't offer any improvement to the predictions.
