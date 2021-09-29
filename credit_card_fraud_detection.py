#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: credit_card_fraud_detection.py
Description: Perform classification on credit card transaction dataset. Part of project for ISYE 6740
Author: Kishan Patel (kpatel605)

"""

# load modules
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, f1_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline



# import data
data = pd.read_csv(r'creditcard.csv')


# identify features and class
X = data[['V1',
 'V2',
 'V3',
 'V4',
 'V5',
 'V6',
 'V7',
 'V8',
 'V9',
 'V10',
 'V11',
 'V12',
 'V13',
 'V14',
 'V15',
 'V16',
 'V17',
 'V18',
 'V19',
 'V20',
 'V21',
 'V22',
 'V23',
 'V24',
 'V25',
 'V26',
 'V27',
 'V28',
 'Amount',]].copy()

y = data.Class.copy()


# separate into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1)


# plot first two pca dimensions of train set
mask_train_fraud = y_train==1
fig, ax = plt.subplots(1,1, figsize=(7,5))
ax.scatter(X_train.loc[~mask_train_fraud, 'V1'], X_train.loc[~mask_train_fraud, 'V2'],
           color='blue', edgecolors='gray', alpha=0.5, label='Legitimate')
ax.scatter(X_train.loc[mask_train_fraud, 'V1'], X_train.loc[mask_train_fraud, 'V2'],
           color='orange', edgecolors='gray', alpha=0.5, label='Fraud')
ax.legend()
ax.set_title('First Two Principal Components for Train Set')
ax.set_xlabel('V1')
ax.set_ylabel('V2')
# plt.show()
plt.savefig('pca_v1v2.png', bbox_inches='tight')

# plot 3rd and 4th pca dimensions
fig, ax = plt.subplots(1,1, figsize=(7,5))
ax.scatter(X_train.loc[~mask_train_fraud, 'V3'], X_train.loc[~mask_train_fraud, 'V4'],
           color='blue', edgecolors='gray', alpha=0.5, label='Legitimate')
ax.scatter(X_train.loc[mask_train_fraud, 'V3'], X_train.loc[mask_train_fraud, 'V4'],
           color='orange', edgecolors='gray', alpha=0.5, label='Fraud')
ax.legend()
ax.set_title('Second and Third Principal Components for Train Set')
ax.set_xlabel('V3')
ax.set_ylabel('V4')
# plt.show()
plt.savefig('pca_v3v4.png', bbox_inches='tight')
plt.close()


# --- PART 1 Basic Classifiers ---

# logistic regression
# fit model
lr = LogisticRegression(random_state=1, max_iter=5000).fit(X_train, y_train)

# get predictions on test set
y_pred_lr = lr.predict(X_test)

# present basic results
print('\n--- Logistic Regression ---')
print(confusion_matrix(y_test, y_pred_lr))
print()
print(classification_report(y_test, y_pred_lr))
print(f'Accuracy: {lr.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_lr = lr.predict_proba(X_test)
probs_lr = probs_lr[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_lr):.4f}')


# compute true positve rate and false positive rate (roc curve)
fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)

# compute precision-recall curve
precision_lr, recall_lr, _ = precision_recall_curve(y_test, probs_lr)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_lr):.4f}')
print(f'Area under PRC: {auc(recall_lr, precision_lr):.4f}')


# neural network
# fit model
mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 10), activation='relu', solver='sgd',
                    random_state=1, max_iter=10000)
mlp.fit(X_train, y_train)

# get predictions on test set
y_pred_mlp= mlp.predict(X_test)

# present basic results
print('\n--- Neural Network ---')
print(confusion_matrix(y_test, y_pred_mlp))
print()
print(classification_report(y_test, y_pred_mlp))
print(f'Accuracy: {mlp.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_mlp = mlp.predict_proba(X_test)
probs_mlp = probs_mlp[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_mlp):.4f}')

# compute true positve rate and false positive rate (roc curve)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, probs_mlp)

# compute precision-recall curve
precision_mlp, recall_mlp, _ = precision_recall_curve(y_test, probs_mlp)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_mlp):.4f}')
print(f'Area under PRC: {auc(recall_mlp, precision_mlp):.4f}')


# classification and regression tree (cart)
# fit model
dt = DecisionTreeClassifier(random_state=1, max_depth=7).fit(X_train, y_train)

# get predictions on test set
y_pred_dt = dt.predict(X_test)

# present basic results
print('\n--- Decision Tree ---')
print(confusion_matrix(y_test, y_pred_dt))
print()
print(classification_report(y_test, y_pred_dt))
print(f'Accuracy: {dt.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_dt = dt.predict_proba(X_test)
probs_dt = probs_dt[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_dt):.4f}')

# compute true positve rate and false positive rate (roc curve)
fpr_dt, tpr_dt, _ = roc_curve(y_test, probs_dt)

# compute precision-recall curve
precision_dt, recall_dt, _ = precision_recall_curve(y_test, probs_dt)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_dt):.4f}')
print(f'Area under PRC: {auc(recall_dt, precision_dt):.4f}')



# plot roc curves
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression, {roc_auc_score(y_test, probs_lr):.4f}')
ax.plot(fpr_mlp, tpr_mlp, label=f'Neural Networks, {roc_auc_score(y_test, probs_mlp):.4f}')
ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree, {roc_auc_score(y_test, probs_dt):.4f}')
ax.set_title('Receiver Operating Characteristic Curves (ROC)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.grid(linestyle='--')
ax.legend()
# plt.show()
plt.savefig('roc.png')
plt.close()

# plot prc curves
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(recall_lr, precision_lr, label=f'Logistic Regression, {auc(recall_lr, precision_lr):.4f}')
ax.plot(recall_mlp, precision_mlp, label=f'Neural Networks, {auc(recall_mlp, precision_mlp):.4f}')
ax.plot(recall_dt, precision_dt, label=f'Decision Tree, {auc(recall_dt, precision_dt):.4f}')
ax.set_title('Precision-Recall Curves (PRC)')
ax.set_xlabel('Recall (True Positive Rate)')
ax.set_ylabel('Precision')
ax.grid(linestyle='--')
ax.legend()
# plt.show()
plt.savefig('prc.png')
plt.close()




# --- PART 2 Improving Classifiers ---

# resample - SMOTE
sm = SMOTE(sampling_strategy='not majority', random_state=1)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
Counter(y_train_res)

# plot first two pca dimensions of smote train set
mask_train_fraud = y_train_res==1
fig, ax = plt.subplots(1,1, figsize=(7,5))
ax.scatter(X_train_res.loc[~mask_train_fraud, 'V1'], X_train_res.loc[~mask_train_fraud, 'V2'],
           color='blue', edgecolors='gray', alpha=0.5, label='Legitimate')
ax.scatter(X_train_res.loc[mask_train_fraud, 'V1'], X_train_res.loc[mask_train_fraud, 'V2'],
           color='orange', edgecolors='gray', alpha=0.5, label='Fraud')
ax.legend()
ax.set_title('First Two Principal Components for SMOTE Resampled Train Set')
ax.set_xlabel('V1')
ax.set_ylabel('V2')
# plt.show()
plt.savefig('pca_v1v2_smote.png', bbox_inches='tight')


# logistic regression
# fit model
lr = LogisticRegression(random_state=1, max_iter=5000).fit(X_train_res, y_train_res)

# get predictions on test set
y_pred_lr = lr.predict(X_test)

# present basic results
print('\n--- Logistic Regression with SMOTE ---')
print(confusion_matrix(y_test, y_pred_lr))
print()
print(classification_report(y_test, y_pred_lr))
print(f'Accuracy: {lr.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_lr = lr.predict_proba(X_test)
probs_lr = probs_lr[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_lr):.4f}')


# compute true positve rate and false positive rate (roc curve)
fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)

# compute precision-recall curve
precision_lr, recall_lr, _ = precision_recall_curve(y_test, probs_lr)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_lr):.4f}')
print(f'Area under PRC: {auc(recall_lr, precision_lr):.4f}')


# neural network
# fit model
mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 10), activation='relu', solver='sgd',
                    random_state=1, max_iter=10000)
mlp.fit(X_train_res, y_train_res)

# get predictions on test set
y_pred_mlp= mlp.predict(X_test)

# present basic results
print('\n--- Neural Network with SMOTE ---')
print(confusion_matrix(y_test, y_pred_mlp))
print()
print(classification_report(y_test, y_pred_mlp))
print(f'Accuracy: {mlp.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_mlp = mlp.predict_proba(X_test)
probs_mlp = probs_mlp[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_mlp):.4f}')

# compute true positve rate and false positive rate (roc curve)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, probs_mlp)

# compute precision-recall curve
precision_mlp, recall_mlp, _ = precision_recall_curve(y_test, probs_mlp)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_mlp):.4f}')
print(f'Area under PRC: {auc(recall_mlp, precision_mlp):.4f}')


# classification and regression tree (cart)
# fit model
dt = DecisionTreeClassifier(random_state=1, max_depth=7).fit(X_train_res, y_train_res)

# get predictions on test set
y_pred_dt = dt.predict(X_test)

# present basic results
print('\n--- Decision Tree with SMOTE ---')
print(confusion_matrix(y_test, y_pred_dt))
print()
print(classification_report(y_test, y_pred_dt))
print(f'Accuracy: {dt.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_dt = dt.predict_proba(X_test)
probs_dt = probs_dt[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_dt):.4f}')

# compute true positve rate and false positive rate (roc curve)
fpr_dt, tpr_dt, _ = roc_curve(y_test, probs_dt)

# compute precision-recall curve
precision_dt, recall_dt, _ = precision_recall_curve(y_test, probs_dt)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_dt):.4f}')
print(f'Area under PRC: {auc(recall_dt, precision_dt):.4f}')


# plot prc curves
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(recall_lr, precision_lr, label=f'Logistic Regression, {auc(recall_lr, precision_lr):.4f}')
ax.plot(recall_mlp, precision_mlp, label=f'Neural Networks, {auc(recall_mlp, precision_mlp):.4f}')
ax.plot(recall_dt, precision_dt, label=f'Decision Tree, {auc(recall_dt, precision_dt):.4f}')
ax.set_title('Precision-Recall Curves (PRC) with SMOTE')
ax.set_xlabel('Recall (True Positive Rate)')
ax.set_ylabel('Precision')
ax.grid(linestyle='--')
ax.legend()
# plt.show()
plt.savefig('prc_smote.png')
plt.close()



# resample - SMOTE + undersampling legitimate charges

over = SMOTE(sampling_strategy=0.1)  # fraud is 10% of legit
under = RandomUnderSampler(sampling_strategy=0.5)  # legit is double fraud

# create pipeline
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# resample
X_train_res2, y_train_res2 = pipeline.fit_resample(X_train, y_train)
Counter(y_train_res2)


# plot first two pca dimensions of smote train set
mask_train_fraud = y_train_res2==1
fig, ax = plt.subplots(1,1, figsize=(7,5))
ax.scatter(X_train_res2.loc[~mask_train_fraud, 'V1'], X_train_res2.loc[~mask_train_fraud, 'V2'],
           color='blue', edgecolors='gray', alpha=0.5, label='Legitimate')
ax.scatter(X_train_res2.loc[mask_train_fraud, 'V1'], X_train_res2.loc[mask_train_fraud, 'V2'],
           color='orange', edgecolors='gray', alpha=0.5, label='Fraud')
ax.legend()
ax.set_title('First Two Principal Components with SMOTE and Undersampling')
ax.set_xlabel('V1')
ax.set_ylabel('V2')
# plt.show()
plt.savefig('pca_v1v2_smote_under.png', bbox_inches='tight')


# logistic regression
# fit model
lr = LogisticRegression(random_state=1, max_iter=5000).fit(X_train_res2, y_train_res2)

# get predictions on test set
y_pred_lr = lr.predict(X_test)

# present basic results
print('\n--- Logistic Regression with SMOTE and Undersampling Majority ---')
print(confusion_matrix(y_test, y_pred_lr))
print()
print(classification_report(y_test, y_pred_lr))
print(f'Accuracy: {lr.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_lr = lr.predict_proba(X_test)
probs_lr = probs_lr[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_lr):.4f}')


# compute true positve rate and false positive rate (roc curve)
fpr_lr, tpr_lr, _ = roc_curve(y_test, probs_lr)

# compute precision-recall curve
precision_lr, recall_lr, _ = precision_recall_curve(y_test, probs_lr)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_lr):.4f}')
print(f'Area under PRC: {auc(recall_lr, precision_lr):.4f}')


# neural network
# fit model
mlp = MLPClassifier(hidden_layer_sizes=(20, 10, 10), activation='relu', solver='sgd',
                    random_state=1, max_iter=10000)
mlp.fit(X_train_res2, y_train_res2)

# get predictions on test set
y_pred_mlp= mlp.predict(X_test)

# present basic results
print('\n--- Neural Network with SMOTE and Undersampling Majority ---')
print(confusion_matrix(y_test, y_pred_mlp))
print()
print(classification_report(y_test, y_pred_mlp))
print(f'Accuracy: {mlp.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_mlp = mlp.predict_proba(X_test)
probs_mlp = probs_mlp[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_mlp):.4f}')

# compute true positve rate and false positive rate (roc curve)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, probs_mlp)

# compute precision-recall curve
precision_mlp, recall_mlp, _ = precision_recall_curve(y_test, probs_mlp)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_mlp):.4f}')
print(f'Area under PRC: {auc(recall_mlp, precision_mlp):.4f}')



# classification and regression tree (cart)
# fit model
dt = DecisionTreeClassifier(random_state=1, max_depth=7).fit(X_train_res2, y_train_res2)

# get predictions on test set
y_pred_dt = dt.predict(X_test)

# present basic results
print('\n--- Decision Tree with SMOTE and Undersampling Majority ---')
print(confusion_matrix(y_test, y_pred_dt))
print()
print(classification_report(y_test, y_pred_dt))
print(f'Accuracy: {dt.score(X_test, y_test):.4f}')

# get probabilities on test set
probs_dt = dt.predict_proba(X_test)
probs_dt = probs_dt[:,1] # keep probabilities for positive outcome (fraud) only

# calculate auc score
print(f'Area under ROC: {roc_auc_score(y_test, probs_dt):.4f}')

# compute true positve rate and false positive rate (roc curve)
fpr_dt, tpr_dt, _ = roc_curve(y_test, probs_dt)

# compute precision-recall curve
precision_dt, recall_dt, _ = precision_recall_curve(y_test, probs_dt)

# compute f1-score and area under precision-recall curve
print(f'F1 score: {f1_score(y_test, y_pred_dt):.4f}')
print(f'Area under PRC: {auc(recall_dt, precision_dt):.4f}')


# plot prc curves
fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(recall_lr, precision_lr, label=f'Logistic Regression, {auc(recall_lr, precision_lr):.4f}')
ax.plot(recall_mlp, precision_mlp, label=f'Neural Networks, {auc(recall_mlp, precision_mlp):.4f}')
ax.plot(recall_dt, precision_dt, label=f'Decision Tree, {auc(recall_dt, precision_dt):.4f}')
ax.set_title('Precision-Recall Curves (PRC) with SMOTE and Undersampling')
ax.set_xlabel('Recall (True Positive Rate)')
ax.set_ylabel('Precision')
ax.grid(linestyle='--')
ax.legend()
# plt.show()
plt.savefig('prc_smote_under.png')
plt.close()
