#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:40:51 2023

@author: galen2
"""
#%%

#RUN THE PROCESSING SCRIPT FIRST!!!

#assuming you followed the instructions above...

random_state = 666

import pandas as pd
import numpy as np

# Load the dataset
file_path = 'FINAL_processed_data.csv'
nmr_data = pd.read_csv(file_path)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import numpy as np

# Preparing the dataset for PLSDA
X = nmr_data.drop(columns=["Sample Number", "Group"])
y = nmr_data["Group"]

# Encode the categorical variable 'Group'
y_encoded = pd.factorize(y)[0]

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state= 42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PLSDA is essentially PLSRegression with a binary response variable
plsda = PLSRegression(n_components= 5)
#MAKE SURE TO ADJUST NUMBER OF COMPONENTS HERE. THSI IS SHOW AT END OF SCRIPT
plsda.fit(X_train_scaled, y_train)

# Predictions and model performance
y_pred_train = plsda.predict(X_train_scaled)
y_pred_test = plsda.predict(X_test_scaled)

# Convert predictions to binary
y_pred_train_binary = np.where(y_pred_train > 0.5, 1, 0)
y_pred_test_binary = np.where(y_pred_test > 0.5, 1, 0)

# Compute metrics
conf_matrix = confusion_matrix(y_test, y_pred_test_binary)
roc_auc = roc_auc_score(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

# Visualization
# Plotting confusion matrix
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax[0].text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
ax[0].set_xlabel('Predictions')
ax[0].set_ylabel('Actuals')
ax[0].set_title('Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_test)
ax[1].plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc:.2f}')
ax[1].plot([0, 1], [0, 1], color='darkblue', linestyle='--')
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic (ROC) Curve')
ax[1].legend()

plt.tight_layout()
plt.show()

#%%

from sklearn.metrics import mean_squared_error

# Calculating Q2 and R2 for the model
def calculate_q2_r2(y_true, y_pred):
    ss_total = sum((y_true - np.mean(y_true)) ** 2)
    ss_res = sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_res / ss_total
    q2 = 1 - mean_squared_error(y_true, y_pred) / np.var(y_true)
    return q2, r2

q2_train, r2_train = calculate_q2_r2(y_train, y_pred_train[:, 0])
q2_test, r2_test = calculate_q2_r2(y_test, y_pred_test[:, 0])


# Identify top 15 variables
# Weights of the first component from the PLS-DA model
plsda_weights = plsda.coef_[:, 0]

# Sorting the weights and getting top 15
top_15_indices = np.argsort(np.abs(plsda_weights))[-15:]
top_15_features = X.columns[top_15_indices]
top_15_weights = plsda_weights[top_15_indices]

# Visualizing the top 15 variables
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_15_features, top_15_weights, color='skyblue')
ax.set_title('Top 15 Variables in PLSDA Model')
ax.set_xlabel('Variable Weight')
plt.show()

# Recalculating Q2 and R2 for the complete dataset
y_pred_complete = plsda.predict(scaler.transform(X))
q2_total, r2_total = calculate_q2_r2(y_encoded, y_pred_complete[:, 0])

# Update visualization for Q2 and R2 with total values
fig, ax = plt.subplots(figsize=(8, 5))
scores = [q2_train, q2_test, r2_train, r2_test, q2_total, r2_total]
labels = ['Train Q2', 'Test Q2', 'Train R2', 'Test R2', 'Total Q2', 'Total R2']
colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']

ax.bar(labels, scores, color=colors)
for i, score in enumerate(scores):
    ax.text(i, score, f'{score:.2f}', ha='center', va='bottom')

ax.set_title('Q2 and R2 Scores (Including Total)')
ax.set_ylabel('Score')
plt.show()


from sklearn.model_selection import KFold

# Number of folds
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Prepare data
X_scaled = scaler.fit_transform(X)
y_encoded = pd.factorize(y)[0]

# Initialize a list to store Q2 values for each fold
q2_scores = []

for train_index, test_index in kf.split(X_scaled):
    # Splitting the data
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]
    
    # Training the model
    plsda = PLSRegression(n_components=5)
    plsda.fit(X_train, y_train)
    
    # Making predictions
    y_pred = plsda.predict(X_test)

    # Calculating Q2
    q2, _ = calculate_q2_r2(y_test, y_pred[:, 0])
    q2_scores.append(q2)
    
# Calculating the average Q2 score
avg_q2_score = np.mean(q2_scores)
print(f"Average cross validated (10 fold) Q2 Score: {avg_q2_score:.2f}")

#%%

import numpy as np
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler # or your preferred scaler
import matplotlib.pyplot as plt

# Assuming X, y_encoded, and calculate_q2_r2 are already defined

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Prepare data
scaler = StandardScaler()  # Replace with your scaler if different
X_scaled = scaler.fit_transform(X)

max_components = 15
q2_scores = []

for n_components in range(1, max_components + 1):
    fold_q2_scores = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        plsda = PLSRegression(n_components=n_components)
        plsda.fit(X_train, y_train)
        y_pred = plsda.predict(X_test)

        q2 = calculate_q2_r2(y_test, y_pred[:, 0])[0]  # Assuming the first return value is Q2
        fold_q2_scores.append(q2)

    avg_q2 = np.mean(fold_q2_scores)
    q2_scores.append(avg_q2)

# Identify the optimal number of components and its corresponding Q2 score
optimal_components = np.argmax(q2_scores) + 1
optimal_q2_score = q2_scores[optimal_components - 1]
print(f"Optimal number of components: {optimal_components}")
print(f"Corresponding Q2 score: {optimal_q2_score:.2f}")

# Plotting
fig, ax = plt.subplots()
components = range(1, max_components + 1)
ax.plot(components, q2_scores, label='Q2', marker='o')

# Highlighting the max value
ax.scatter(optimal_components, optimal_q2_score, color='red', zorder=5)

ax.set_xlabel('Number of Components')
ax.set_ylabel('Q2 Score')
ax.set_title('Q2 Scores vs Number of Components')
ax.legend()
plt.show()


#%%

def calculate_vip_scores(pls_model, X, y):
    t = pls_model.x_scores_  # Scores
    w = pls_model.x_weights_  # Weights
    q = pls_model.y_loadings_  # Y Loadings

    p, h = w.shape
    vips = np.zeros((p,))

    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)

    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) * np.sqrt(s[j]) for j in range(h)])
        vips[i] = np.sqrt(p * (weight.T @ weight) / total_s)

    return vips

vip_scores = calculate_vip_scores(plsda, X_train_scaled, y_train)

def plot_vip_scores(vip_scores, feature_names):
    sorted_indices = np.argsort(vip_scores)[::-1]
    sorted_vip_scores = vip_scores[sorted_indices]
    sorted_features = feature_names[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Highlight bars above the threshold
    above_threshold = sorted_vip_scores >= 1.0
    ax.bar(range(len(sorted_vip_scores)), sorted_vip_scores, align='center', 
           color=np.where(above_threshold, 'red', 'skyblue'))

    ax.set_xticks(range(len(sorted_vip_scores)), sorted_features, rotation='vertical')
    ax.set_xlabel('Features')
    ax.set_ylabel('VIP Scores')
    ax.set_title('Variable Importance in Projection (VIP) Scores')

    # Draw a horizontal line at VIP = 1.0
    ax.axhline(y=1.0, color='green', linestyle='--')

    plt.tight_layout()
    plt.show()

plot_vip_scores(vip_scores, np.array(X.columns))


def plot_vip_scores_filtered(vip_scores, feature_names):
    # Filter features with VIP scores greater than 1.0
    filtered_indices = vip_scores >= 1.0
    filtered_vip_scores = vip_scores[filtered_indices]
    filtered_features = feature_names[filtered_indices]

    # Sort the filtered scores and features in descending order
    sorted_indices = np.argsort(filtered_vip_scores)[::-1]
    sorted_filtered_vip_scores = filtered_vip_scores[sorted_indices]
    sorted_filtered_features = filtered_features[sorted_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create horizontal bar plot for features with VIP > 1.0
    ax.barh(range(len(sorted_filtered_vip_scores)), sorted_filtered_vip_scores, align='center', color='red')
    ax.set_yticks(range(len(sorted_filtered_vip_scores)))
    ax.set_yticklabels(sorted_filtered_features)

    ax.set_ylabel('Features')
    ax.set_xlabel('VIP Scores')
    ax.set_title('Variable Importance in Projection (VIP) Scores (VIP > 1.0)')

    # Reverse the vertical axis to display the highest scores at the top
    ax.invert_yaxis()

    plt.tight_layout()
    plt.show()

plot_vip_scores_filtered(vip_scores, np.array(X.columns))
#%%
# this  will loop through each feature with a VIP score greater than 1.0 and create a separate boxplot for each, 
#saving them as individual image files into the wd
import seaborn as sns
import matplotlib.pyplot as plt

def save_individual_vip_boxplots(vip_scores, features, data, group_label, save_path, palette):
    # Identify features with VIP scores > 1.0
    high_vip_indices = vip_scores > 1.0
    high_vip_features = features[high_vip_indices]

    # Create and save a boxplot for each feature
    for feature in high_vip_features:
        # Extract data for this feature
        feature_data = data[[feature, group_label]]

        # Create the boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=group_label, y=feature, data=feature_data, palette=palette)
        plt.title(f'Boxplot of {feature}')
        plt.xlabel('Group')
        plt.ylabel(feature)

        # Save the plot
        plt.savefig(f'{feature}_boxplot.png')
        #adjust this to deal with icloud
        plt.close()

# Example usage
# Assuming 'X' is your dataset with features, 'y' is the group label, and 'np.array(X.columns)' are the feature names
# Replace 'path/to/save' with your desired save path
# Define your color palette, e.g., ['blue', 'orange']
color_palette = ['purple', 'orange']
save_individual_vip_boxplots(vip_scores, np.array(X.columns), X.join(y), 'Group', 'path/to/save', color_palette)


#%%
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def save_individual_vip_boxplots_with_stats(vip_scores, features, data, group_label, save_path, palette):
    # Identify features with VIP scores > 1.0
    high_vip_indices = vip_scores > 1.0
    high_vip_features = features[high_vip_indices]

    # Create and save a boxplot for each feature
    for feature in high_vip_features:
        # Extract data for this feature
        feature_data = data[[feature, group_label]].dropna()

        # Create the boxplot
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=group_label, y=feature, data=feature_data, palette=palette)
        plt.title(f'Boxplot of {feature}')
        plt.xlabel('Group')
        plt.ylabel(feature)

        # Extract groups
        groups = feature_data[group_label].unique()
        group_data = [feature_data[feature_data[group_label] == g][feature] for g in groups]

        # Perform the Mann-Whitney U test if both groups have data
        if all(len(g) > 0 for g in group_data) and len(groups) == 2:
            stat, p_value = mannwhitneyu(*group_data)
            plt.text(0.5, 0.1, f'p-value = {p_value:.2e}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        else:
            plt.text(0.5, 0.1, 'Stat test not applicable', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

        # Save the plot
        plt.savefig(f'{feature}_boxplot_w_signif.png')
        plt.close()

# Example usage
color_palette = ['blue', 'orange']
save_individual_vip_boxplots_with_stats(vip_scores, np.array(X.columns), X.join(y), 'Group', 'path/to/save', color_palette)
