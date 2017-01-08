'''
This is a modified version of a script written by Danijel Kivaranovic at:
https://www.kaggle.com/danijelk/allstate-claims-severity/keras-starter-with-bagging-lb-1120-596

LightGBM model to generate test and out of fold train predictions for the
kaggle Allstate claims severity competition.

input: Raw CSV supplied by Allstate
output: Test and out of fold train predictions in a CSV
'''

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from pylightgbm.models import GBMRegressor
import random

##Input Constants
nfolds = 5
nbags= 20
SEED=2016
shift = 200
path_to_exec= "~/LightGBM/lightgbm"

random.seed(SEED)


## read data
train = pd.read_csv('~/Desktop/kaggle/AllState/input/train.csv')
test = pd.read_csv('~/Desktop/kaggle/AllState/input/test.csv')

## set test loss to NaN
test['loss'] = np.nan

## response and IDs
y = np.log(train['loss'].values + shift)
id_train = train['id'].values
id_test = test['id'].values

## stack train test
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis=0)

## Preprocessing and transforming to sparse data
sparse_data = []

f_cat = [f for f in tr_te.columns if 'cat' in f]
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)

f_num = [f for f in tr_te.columns if 'cont' in f]
scaler = StandardScaler()
tmp = csr_matrix(scaler.fit_transform(tr_te[f_num]))
sparse_data.append(tmp)

del (tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format='csr')
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

del (xtr_te, sparse_data, tmp)

## cv-folds
folds = KFold(len(y), n_folds=nfolds, shuffle=True, random_state=111)

## train models
i = 0
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])



for (inTr, inTe) in folds:
    print 'Fold: ' + str(i) + " Training..."
    print type(inTr)
    print type(xtrain)
    x_tr = xtrain[inTr]
    y_tr = y[inTr]
    x_val = xtrain[inTe]
    y_val = y[inTe]
    pred = np.zeros(x_val.shape[0])

    for j in range(nbags):
        print 'Bag: ' + str(j)
        rand_seed = random.randint(1, 5000)
        gbmr = GBMRegressor(
            exec_path=path_to_exec,  # Change this to your LighGBM path
            config='',
            application='regression',
            num_iterations=int(2558/0.9),
            learning_rate=0.01,
            num_leaves=200,
            num_threads=4,
            min_data_in_leaf=8,
            metric='l1',
            feature_fraction=0.3,
            feature_fraction_seed=rand_seed,
            bagging_fraction=0.8,
            bagging_freq=100,
            bagging_seed=rand_seed,
            verbose=False
        )
        # Train
        gbmr.fit(x_tr, y_tr, test_data=[(x_val, y_val)])

        # Apply to validation and test data
        print 'Bag: ' + str(j) + " Predicting..."
        pred += np.exp((gbmr.predict(x_val))) - shift
        pred_test += np.exp((gbmr.predict(xtest))) - shift

    # Save oob results
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(np.exp(y_val) - shift, pred)
    print 'Fold ' + str(i) + '- MAE:' + str(score)
    i += 1


# Get mean of pred_test
pred_test /= (nfolds * nbags)

## train predictions
df = pd.DataFrame({'loss': pred_oob})
df.to_csv('LightGBM2_preds_oob.csv', index=False)

## test predictions
df = pd.DataFrame({'id': id_test, 'loss': pred_test})
df.to_csv('LightGBM2_submission_shift_perm.csv', index=False)
