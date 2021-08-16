import numpy as np
import matplotlib.pyplot as plt
import os 
import pickle
import argparse
from scipy import stats


parser = argparse.ArgumentParser()
parser.add_argument("--ROC-per-label-path-embeddings", dest='ROC_per_label_path_embeddings',
                    help='full path to the npy file that contains an array representing the AUC ROC per label for the model with word embeddings')
parser.add_argument("--PR-per-label-path-embeddings", dest='PR_per_label_path_embeddings',
                    help='full path to the npy file that contains an array representing the AUC PR per label for the model with word embeddings')
parser.add_argument("--ROC-per-label-path-baseline", dest='ROC_per_label_path_baseline',
                    help='full path to the npy file that contains an array representing the AUC ROC per label for the baseline model')
parser.add_argument("--PR-per-label-path-baseline", dest='PR_per_label_path_baseline',
                    help='full path to the npy file that contains an array representing the AUC PR per label for the baseline model')
parser.add_argument("--list-tags-baseline", dest='list_tags_baseline',
                    help='full path to the npy file that contains an array containing the names of the tags for the baseline model')
args = parser.parse_args()

"""
Model with wwords embeddings
"""

PR_embeddings=np.load(args.PR_per_label_path_embeddings)
ROC_embeddings=np.load(args.ROC_per_label_path_embeddings)

plt.hist(PR_embeddings[-1,:], histtype='bar', ec='black')
plt.title('Histogram of AUC PR - word embeddings')
kde = stats.gaussian_kde(PR_embeddings[-1,:])
xx = np.linspace(0, 0.8, 1000)
plt.plot(xx, kde(xx))
plt.xlim([0.0,0.7])
plt.ylim([0.0,9])

plt.xlabel('AUC PR')
plt.legend(['KDE','counts'])
plt.show()
plt.show()

PR_sorted_clusters_embeddings = sorted(list(range(len(PR_embeddings[-1,:]))), key=lambda k: PR_embeddings[-1,k]) 
print('The clusters sorted by performance in PR of the model with word embeddings are \n',PR_sorted_clusters_embeddings)
PR_Final_embeddings=np.sort(PR_embeddings[-1,:])
print('The final PR is',np.mean(PR_Final_embeddings))

ROC_sorted_clusters_embeddings = sorted(range(len(ROC_embeddings[-1,:])), key=lambda k: ROC_embeddings[-1,k]) 
print('The clusters sorted by performance in ROC of the model with word embeddings  are \n',ROC_sorted_clusters_embeddings)

"""
Baseline model
"""

PR=np.load(args.PR_per_label_path_baseline)
ROC=np.load(args.ROC_per_label_path_baseline)


PR_sorted_clusters = sorted(list(range(len(PR[-1,:]))), key=lambda k: PR[-1,k]) 
print('The clusters sorted by performance in PR of the baseline model are \n',PR_sorted_clusters)
PR_Final=np.sort(PR[-1,:])
print('The final PR is',np.mean(PR_Final))

ROC_sorted_clusters = sorted(range(len(ROC[-1,:])), key=lambda k: ROC[-1,k]) 
print('The clusters sorted by performance in ROC of the baseline model are \n',ROC_sorted_clusters)


plt.hist(PR[-1,:], histtype='bar', ec='black')
plt.title('Normalized histogram of AUC PR - baseline model')
kde = stats.gaussian_kde(PR[-1,:])
xx = np.linspace(0, 0.8, 1000)
plt.plot(xx, kde(xx))
plt.xlim([0.0,0.7])
plt.ylim([0.0,9])

plt.xlabel('AUC PR')
plt.legend(['KDE','normalized counts'])
plt.show()
plt.show()

