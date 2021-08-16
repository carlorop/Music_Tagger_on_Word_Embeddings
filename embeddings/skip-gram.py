"""Contains tools for creating the clusters based on tags embeddings"""

import json
import os
import pickle
from itertools import compress
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from k_means_normalized import k_means
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import normalize
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

parser = argparse.ArgumentParser()
parser.add_argument("--save-path", dest='save_path',
                    help='path in which we save the dictionary of clusters')
parser.add_argument("--genre-path", dest='genre_path',
                    help='path of the directory that contains genres.txt')
parser.add_argument("--samples-path", dest='samples_path',
                    help='directory which contains the set of encoded vectors for each genre, the name of the files must be "genre_samples_without_main_"+index of genre+" ".npy"')
parser.add_argument("--genre-list-path", dest='genre_list_path',
                    help='directory which contains the list of subgenres of each genre , the name of the files must be "genre_"+index of genre+" ".npy"')
args = parser.parse_args()


def plot_silhuette(X, predictions, genre):
    n_clusters = len(np.unique(predictions))
    y = predictions.copy()
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    for i, j in enumerate(np.unique(y)):
        y[predictions == j] = i

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(X, y)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[y == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title('Silhouette plot for genre "' + genre + '"')
    ax1.set_xlabel("Silhouette coefficient")
    ax1.set_ylabel("Clusters")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.grid()
    plt.show()


def parse_input(data):
    x = []
    y = []
    _, dimension = (data.shape)
    for row in data:
        num_labels = np.sum(row)
        input_tensor = np.zeros((num_labels, dimension), dtype=np.int32)
        output_tensor = np.array([row, ] * num_labels)
        k = 0
        true_labels = list(compress(range(len(row)), row))
        for k, i in enumerate(true_labels):
            input_tensor[k, i] = 1
            output_tensor[k, i] = 0
        x.append(input_tensor)
        y.append(output_tensor)
    return np.vstack(x), np.vstack(y), dimension


def my_loss_fn(y_true, y_pred):
    probabilites = []
    for i in range(len(y_true)):
        temp = y_true[i] * y_pred[i]
        if temp != 0:
            probabilites.append(temp)
    return -np.log(np.prod(probabilites))


clusters_dictionary = {}
cumulative_clusters = 0
save_path = args.save_path
correspondence_genres = {1: 'soul', 3: 'pop', 13: 'electronic', 22: 'Trance', 23: 'blues', 24: 'rock', 25: 'jazz',
                         27: 'indie', 34: 'Hip-Hop', 37: 'heavy metal', 38: 'house'}

with open(os.path.join(args.genre_path, 'genres.txt'), "rb") as f:
    genres, _ = f.readlines()
    genres = np.sort(json.loads(genres))

for genre in genres:
    try:
        # The list of subgenres includes the main genre
        genre_list = np.load(args.genre_list_path)[1:]

        with open(args.samples_path, 'rb') as f:
            data = np.load(f)

        x, y, dimension = parse_input(data)
        model = Sequential()
        model.add(Dense(np.floor(np.sqrt(dimension)), input_dim=dimension))
        model.add(Dense(dimension, activation=softmax))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

        model.fit(x, y, epochs=50, batch_size=10, verbose=1, callbacks=[callback])
        embeddings = normalize(model.layers[0].get_weights()[0])

        record_silhuette = []
        min_clusters = 5
        max_clusters = 20

        for i in range(min_clusters, max_clusters):
            predictions, _, _ = k_means(embeddings, i, 60)
            l = silhouette_score(embeddings, predictions)
            record_silhuette.append(l)
            print('\n number of clusters:', i, ' ', l, end=', ')
            for x in np.unique(predictions):
                print(list(predictions.flatten()).count(x), end=', ')

        max_silhouette = max(record_silhuette)
        optimum_cluster = record_silhuette.index(max_silhouette) + min_clusters
        predictions, _, _ = k_means(embeddings, optimum_cluster, 60)
        predictions += cumulative_clusters
        print('\n number of clusters:', optimum_cluster, ', silhuette score: ', max_silhouette, end=', ')

        # Remove the labels that are included in clusters with just one tag
        for x in np.unique(predictions):
            counts = list(predictions.flatten()).count(x)
            if counts == 1:
                index_remove = np.where(predictions == x)[0]
                predictions = np.delete(predictions, index_remove)
                genre_list = np.delete(genre_list, index_remove)
                embeddings = np.delete(embeddings, index_remove, 0)
                continue
            print(counts, end=', ')

        plot_silhuette(embeddings, predictions, correspondence_genres[genre])

        genre_dictionary = dict(zip(genre_list, predictions))
        clusters_dictionary = dict(clusters_dictionary, **genre_dictionary)
        cumulative_clusters += optimum_cluster
    except:
        pass

print('We have obtained ', len(np.unique(list(clusters_dictionary.values()))), ' clusters')
with open(os.path.join(save_path, 'tags_dict_500.pkl'), "wb") as f:
    pickle.dump(clusters_dictionary, f)
