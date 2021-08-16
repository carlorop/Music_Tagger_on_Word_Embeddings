"""
Contains different data inspections related to the co-occurence matrix and the results of the classifier
"""

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--matrix-path", dest='matrix_path',
                    help='path to the file that contains the coocurrence matrix, saved as a .npy file')
parser.add_argument("--dictionary-path", dest='dictionary_path',
                    help='full path of the pickle file that contains the dictionary of clusters')
parser.add_argument("--save-genre-path", dest='save_genre_path',
                    help='path of the directory in which we save genres.txt')
parser.add_argument("--ROC-per-label-path", dest='ROC_per_label_path',
                    help='full path to the npy file that contains an array representing the AUC ROC per label')
parser.add_argument("--PR-per-label-path", dest='PR_per_label_path',
                    help='full path to the npy file that contains an array representing the AUC PR per label')
args = parser.parse_args()

b = np.load(args.matrix_path)
b_diag = np.diag(b).astype('int64')

# We normalize the co-ocurence matrix because popular tags lead to large co-ocurences
b_normalized = 1. * b
for i in range(len(b_diag)):
    for j in range(len(b_diag)):
        b_normalized[i, j] /= np.sqrt(b_diag[i] * b_diag[j])

correlated_clusters = [i for i in np.argwhere(b_normalized > 0.5) if i[0] != i[1]]
m = 1.5
# This variable will store the pairs of clusters in which the less frequent cluster co-occurs with the largest cluster in at least 1/m of the occurences of the largest clsuter
correlated_clusters_m = []
for i in range(len(b)):
    for j in range(len(b.T)):
        if i != j:
            if b_normalized[i, j] > np.sqrt(np.minimum(b[i, i] / b[j, j], b[j, j] / b[i, i])) / m:
                correlated_clusters_m.append([i, j])

with open(args.dictionary_path, "rb") as f:
    dictionary_clusters = pickle.load(f)

inv_map = {}  # we compute the reversed dictionary
clusters = np.unique(list(dictionary_clusters.values()))  # clusters is a sorted list
converter = {}
for i in range(len(clusters)):
    converter[clusters[i]] = i

for k, v in dictionary_clusters.items():
    inv_map[converter[v]] = inv_map.get(converter[v], []) + [str(k)]

if args.save_genre_path:
    """
    Here we write genres.txt, a file that contains the indices of the clusters that contain genres
    """

    write_genres = True  # False
    np_correlated_clusters_m = np.array(correlated_clusters_m)
    np_correlated_clusters_m.sort(axis=1)
    correlated_clusters_m = [pair.tolist() for pair in np.unique(np_correlated_clusters_m, axis=0)]
    if write_genres:
        genre_path = 'C:/Users/Administrador/Desktop'
        with open(os.path.join(genre_path, 'genres2.txt'), "w") as f:
            f.write(
                '[]\n')  # we will fill the tuple of genres by hand, based on the dictionary inv_map, use its keys as clusters
            f.write(str((correlated_clusters_m)))

if args.ROC_per_label_path:
    ROC = np.load(args.ROC_per_label_path)
    ROC_sorted_clusters = sorted(list(range(len(ROC[-1, :]))), key=lambda k: ROC[-1, k])
    print('The clusters sorted by performance using the AUC ROC are \n', ROC_sorted_clusters)
    ROC_Final = np.sort(ROC[-1, :])
    print('The final AUC ROC is', np.mean(ROC_Final))
    plt.hist(ROC[-1, :], histtype='bar', ec='black')
    plt.title('Histogram of AUC ROC')
    plt.show()

if args.PR_per_label_path:
    PR = np.load(args.PR_per_label_path)
    PR_sorted_clusters = sorted(list(range(len(PR[-1, :]))), key=lambda k: PR[-1, k])
    print('The clusters sorted by performance using the AUC PR are \n', PR_sorted_clusters)
    PR_Final = np.sort(PR[-1, :])
    print('The final AUC PR is', np.mean(PR_Final))
    plt.hist(PR[-1, :], histtype='bar', ec='black')
    plt.title('Histogram of AUC PR')
    plt.show()
