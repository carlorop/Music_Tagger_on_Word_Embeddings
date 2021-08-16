"""Contains tools for creating the clusters based on NPMI"""

import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser()
parser.add_argument("--save-path", dest='save_path',
                    help='path in which we save the dictionary of clusters')
parser.add_argument("--genre-path", dest='genre_path',
                    help='path of the directory that contains genres.txt')
parser.add_argument("--matrix-path", dest='matrix_path',
                    help='directory which contains the co-occurrence matrices, the name of the files must be "genre_co-occurrence_"+index of genre+" ".npy"')
parser.add_argument("--genre-list-path", dest='genre_list_path',
                    help='directory which contains the lists of subgenres of each genre , the name of the files must be "genre_"+index of genre+" ".npy"')
args = parser.parse_args()

with open(os.path.join(args.genre_path, 'genres.txt'), "rb") as f:
    genres, _ = f.readlines()
    genres = np.sort(json.loads(genres))

clusters_dictionary = {}
cumulative_clusters = 0
save_path = args.save_path

correspondence_genres = {1: 'soul', 3: 'pop', 13: 'electronic', 22: 'Trance', 23: 'blues', 24: 'rock', 25: 'jazz',
                         27: 'indie', 34: 'Hip-Hop', 37: 'heavy metal', 38: 'house'}

for genre in genres:
    try:  # We handle the exception caused from removing clusters after merging them
        matrix_path = args.matrix_path
        genre_list = np.load(args.genre_list_path)

        b = np.load(matrix_path).astype(np.int64)

        # Basic data filtering, we remove the tags with an insignificant number of counts
        zero_rows = np.where(np.diagonal(b) < np.max(b) ** 0.25)[0]
        b = np.delete(b, zero_rows, 0)
        b = np.delete(b, zero_rows, 1)
        genre_list = np.delete(genre_list, zero_rows)

        # Remove the main genre from the list of subgenres
        b_extended = b.copy()
        main_genre = np.where(np.diagonal(b) == np.max(b))[0]
        b = np.delete(b, main_genre, 0)
        b = np.delete(b, main_genre, 1)
        genre_list = np.delete(genre_list, main_genre)

        p = 0. * b
        p_extended = 0. * b_extended
        N = np.sum(b)
        N_extended = np.sum(b_extended)

        for i in range(len(p_extended)):
            for j in range(len(p_extended.T)):
                temp_extended = 1 - ((np.log2(b_extended[i, j] / (b_extended[i, i] * b_extended[j, j])) + np.log2(
                    N_extended)) / (-np.log2(b_extended[i, j] / N_extended)))
                if temp_extended == temp_extended:  # if temp is nan the identity is False
                    p_extended[i, j] = temp_extended
                    if i > 0 and j > 0:
                        p[i - 1, j - 1] = temp_extended
                else:
                    p_extended[i, j] = 2
                    if i > 0 and j > 0:
                        p[i - 1, j - 1] = 2

        # Fill diagonal with 0 to correct precision errors
        np.fill_diagonal(p, 0)
        np.fill_diagonal(p_extended, 0)

        record_silhuette = []
        min_clusters = 5
        max_clusters = 20

        for i in range(min_clusters, max_clusters):
            agg = AgglomerativeClustering(n_clusters=i, affinity='precomputed',
                                          linkage='average')

            predictions = agg.fit_predict(p)
            # print(u)
            from sklearn.metrics import silhouette_score

            l = silhouette_score(p, predictions, metric="precomputed")
            record_silhuette.append(l)
            print('\n number of clusters:', i, ' ', l, end=', ')
            for x in np.unique(predictions):
                print(list(predictions.flatten()).count(x), end=', ')

        max_silhouette = max(record_silhuette)
        optimum_cluster = record_silhuette.index(max_silhouette) + min_clusters
        agg = AgglomerativeClustering(n_clusters=optimum_cluster, affinity='precomputed',
                                      linkage='average')

        predictions = agg.fit_predict(p) + cumulative_clusters
        print('\n number of clusters:', optimum_cluster, ', silhuette score: ', max_silhouette, end=', ')

        for x in np.unique(predictions):
            counts = list(predictions.flatten()).count(x)
            if counts == 1:
                index_remove = np.where(predictions == x)[0]
                predictions = np.delete(predictions, index_remove)
                genre_list = np.delete(genre_list, index_remove)
                continue
            print(counts, end=', ')

        genre_dictionary = dict(zip(genre_list, predictions))
        clusters_dictionary = dict(clusters_dictionary, **genre_dictionary)
        cumulative_clusters += optimum_cluster
        for x in np.unique(predictions):
            print(list(predictions.flatten()).count(x), end=', ')

        np.random.seed(110)
        X_embedded = TSNE(n_components=2, metric="precomputed").fit_transform(p_extended)
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1])
        plt.scatter(X_embedded[main_genre, 0], X_embedded[main_genre, 1], c='r')
        plt.title('Dimensionality reduction of genre "' + str(correspondence_genres[genre]) + '"')
        plt.show()
    except:
        pass

print('We have obtained ', len(np.unique(list(clusters_dictionary.values()))), ' clusters')
with open(os.path.join(save_path, 'tags_dict_500.pkl'), "wb") as f:
    pickle.dump(clusters_dictionary, f)
