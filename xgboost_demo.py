import numpy as np
import xgboost as xgb

n = 50 # number of datapoints
d = 200 # number of features
X = np.random.rand(n, d) # randomly generate dataset
number_train_classes = 2

train_frac = .3 # fraction of data to be used for training
n_train = int(train_frac * n) # number of training datapoints

# set aside training/validation data, as well as labels
random_indices = np.random.choice(a=list(range(0, n)), size=n_train, replace=False)
X_train = X[0:n_train]
X_valid = X[n_train:]

y_label = np.random.choice(np.arange(number_train_classes), size=n, replace=True)
y_train = y_label[0:n_train]
y_valid = y_label[n_train:]

# randomly create gene names
gene_names = [f"gene_{i}" for i in range(d)]

train_matrix = xgb.DMatrix(data=X_train, label=y_train, feature_names=gene_names)

# defining hyperparameters
eta = .3
xgb_params_train = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class': number_train_classes, # look for class labels {0, 1,..., number_train_classes - 1}
    'eta': eta, # learning rate
    'max_depth': 4,
    'subsample': .6
}
n_round = 10 # number of boosted rounds

# train model
bst_model_train = xgb.train(params=xgb_params_train, dtrain=train_matrix, num_boost_round=n_round)

validation_matrix = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=gene_names)

validation_probs = bst_model_train.predict(data=validation_matrix)

y_pred = np.argmax(validation_probs, axis=1)
accuracy = np.sum(y_pred==y_valid) / len(y_pred)
print(accuracy)