import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from k_means_normalized import k_means, predict_k_means
from lastfm import LastFm
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('algorithm', choices=['word2vec', 'glove'])
parser.add_argument('--glove-embeddings-path', dest='glove_embeddings_path',
                    help='full path to txt file encoding the glove pre-trained embeddings, these embeddings can be downloaded from https://nlp.stanford.edu/projects/glove/')
parser.add_argument("--threshold", dest='threshold',
                    help='specify the threshold, we remove all teh tags whose number of counts in the last.fm dataset is less than the threshold',
                    type=int, default=500)
parser.add_argument("--n-clusters", dest='n_clusters', help='specify the number of clusters',
                    type=int, default=500)
parser.add_argument("--save-path", dest='save_path',
                    help='directory in which we save the dictionary of clusters and the k-means model')
parser.add_argument("--lastfm", help='full path to the lastfm tags database')

args = parser.parse_args()

if args.algorithm == 'word2vec':
    embed = hub.load("https://tfhub.dev/google/Wiki-words-250-with-normalization/2")
else:
    assert args.glove_embeddings_path[-4:] == '.txt', 'Please, provide a valid txt file'
    embeddings_dict = {}
    with open(args.glove_embeddings_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            vector /= np.linalg.norm(vector)
            embeddings_dict[word] = vector


    def embed(sentences):
        """
        Return the embeddings of 'sentence' based on 'embeddings_dict'

        sentences: array ir list 
            iterable containing the words for embedding as str
        """

        assert isinstance(sentences, list) or isinstance(sentences, np.ndarray), 'Please provide a list as input'
        sentences_embedded = []
        for sentence in sentences:
            # we remove unwanted characters such as ? ! . ,
            sentence = sentence.replace('.', '')
            sentence = sentence.replace('!', '')
            sentence = sentence.replace('?', '')
            sentence = sentence.replace(',', '')
            sentence = sentence.lower()  # The words in the corpus are lowercase

            sentence = sentence.split(' ')
            sentence_embedded = []
            for word in sentence:
                try:
                    sentence_embedded.append(embeddings_dict[word])
                except:
                    sentence_embedded.append(
                        np.zeros(200, dtype='float32'))  # See that we are using embeddings with length 200
            if np.any(sentence_embedded):  # check if there are at least an element different from zero
                sentence_embedded = np.sum(sentence_embedded, axis=0)
                sentence_embedded /= np.linalg.norm(sentence_embedded)
                sentences_embedded.append(sentence_embedded)
            else:
                sentences_embedded.append(sentence_embedded[0])
        return tf.constant(sentences_embedded)

n = args.threshold
fm = LastFm(args.lastfm)
tags = fm.popularity()
tags = tags[tags['count'] > n]['tag'].tolist()

old_tags = []
new_tags = []
dict_decade = {'00s': 'Noughties', '10s': 'Teens', '20s': 'Twenties', '30s': 'Thirties', '40s': 'Forties',
               '50s': 'Fifties', '60s': 'Sixties', '70s': 'Seventies', '80s': 'Eighties', '90s': 'Nineties'}
decades = dict_decade.keys()
decades_expand = ['19' + n for n in list(decades)]
decades_expand_2000 = ['20' + n for n in list(decades)]
cummulative = False  # This bolean is useful for tags that require filtering in more than one word
for l, i in enumerate(tags):
    s = (i.split(' '))
    for j, k in enumerate(s):
        if k in decades or k in decades_expand or k in decades_expand_2000:
            if not cummulative:
                old_tags.append(" ".join(s))
            s[j] = dict_decade[k[-3:]]
            cummulative = True
    if cummulative:
        new_tags.append(" ".join(s))
        tags[l] = new_tags[-1]
        cummulative = False

tags_embeddings = embed(tags).numpy()

zero_rows = np.where(~tags_embeddings.any(axis=1))[0]
tags_embeddings = np.delete(tags_embeddings, zero_rows, 0)  # delete the rows 'zero_rows' from axis 0
tags_clean = np.array([k for i, k in enumerate(tags) if i not in zero_rows])  # we create a list of clean tags
for i, j in enumerate(new_tags):
    tags_clean[tags_clean == j] = old_tags[i]

"""
k-means
"""

tags_100 = tags[:100]
path = 'C:/Users/Administrador/Desktop'
file = 'tags_remove.txt'

with open(os.path.join(path, file), 'r') as f:
    tags_remove = f.read().split('\n')
    for i in tags_remove:
        if i in tags_100:
            tags_100.remove(i)

model_path = args.save_path


def filter_clusters(clusters, values, tags_subset):
    """
    Returns the index of the samples of the dataset that are included in the same cluster as the samples of tags_subset
    
    clusters: list
        Contains the list of clusters obtained by running the k means algorithm on the dataset
        
    values: list
        Contains the label of cluster that contains each sample
    
    tags_subset: list
        Subsets of sampls used for filtering 
        
    """
    values_attach = set(predict_k_means(np.array(embed(tags_subset)), clusters))
    index_attach = []
    for i, j in enumerate(values):
        if j in values_attach:
            index_attach.append(i)
    return index_attach


n_clusters = args.n_clusters
print('Starting processing for ' + str(n_clusters) + 'clusters')
values_embeddings, _, clusters = k_means(tags_embeddings, n_clusters, 60)
with open(os.path.join(args.save_path, 'kmeans_' + str(n_clusters) + '.pkl'), "wb") as f:
    pickle.dump(clusters, f)
values = np.array(values_embeddings)
index_attach = filter_clusters(clusters, values, tags_100)
values_attach = values[index_attach]
tags_attach = tags_clean[index_attach]
zip_pairs = zip(tags_attach, values_attach)
tags_dictionary = dict(zip_pairs)
with open(os.path.join(model_path, 'tags_dict_' + str(n_clusters) + '.pkl'), "wb") as f:
    pickle.dump(args.save_path, f)

"""
Inspection of linear substructures of the embeddings
"""

# See that the position of the labels of the points is prepaired for the embeddings using word2Vec
pca = PCA(n_components=2, svd_solver='full')
words = ["Japan", "Tokyo", 'France', 'Paris', 'Germany', 'Berlin', 'Portugal', 'Lisbon', 'Spain', 'Madrid', 'Austria',
         'Vienna']  # ,'Kenya','Nairobi']
vectors = [embed([word]) for word in words]
np.random.seed(1)
Y = pca.fit_transform(embed(words))
plt.scatter(Y[:, 0], Y[:, 1])
for label, x, y in zip(words, Y[:, 0], Y[:, 1]):
    if label == "Japan" or label == "Tokyo":
        plt.annotate(label, xy=(x, y), xytext=(4, -7), textcoords="offset points")
    elif label == 'Portugal':
        plt.annotate(label, xy=(x, y), xytext=(-26, 10), textcoords="offset points")
    elif label == 'Lisbon':
        plt.annotate(label, xy=(x, y), xytext=(0, -10), textcoords="offset points")
    elif label == 'Spain':
        plt.annotate(label, xy=(x, y), xytext=(-20, 5), textcoords="offset points")
    elif label == 'Madrid':
        plt.annotate(label, xy=(x, y), xytext=(-20, -17), textcoords="offset points")
    else:
        plt.annotate(label, xy=(x, y), xytext=(4, 0), textcoords="offset points")

plt.title('Linear substructures in word embeddings')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.grid()
plt.show()
