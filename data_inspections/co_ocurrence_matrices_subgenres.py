"""
In this scripts we create the co-occurence matrix for the subgenres of the main musical genres
"""

import argparse
import json
import os
import pickle
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--tfrecords-dir', dest='tfrecords_dir',
                    help='directory to read the .tfrecord files from ')
parser.add_argument('--dictionary-dir', dest='dictionary_dir',
                    help=' Full path to the dictionary that links the tags with the clusters')
parser.add_argument('--genre-path', dest='genre_path',
                    help='path to the directory which containts genres.txt, a file containing the clusters with genres and the clusters to merge them, the txt is composed by two lines, the first one is a list of clusters [x,..,z] and the second one is a list of pairs of clusters to be merged [[x,y],...]')
parser.add_argument('--save-co-occurence-matrix-path', dest='save_co_occurence_matrix_path',
                    help='path to save an array containing the co-occurence matrix of each genre')
parser.add_argument('--save-tag_vectors-path', dest='save_tag_vectors_path',
                    help='path to save an array containing the co-occurence matrix of each genre')
parser.add_argument('--save-tags-path', dest='save_tags_path',
                    help='path to save an array containing the subgenres for interpreting each component of the classification')

args = parser.parse_args()

tfrecords = []
path_dict = args.dictionary_dir
tfrecords_dir = args.tfrecords_dir
filter_genres_path = args.genre_path

for file in os.listdir(os.path.expanduser(tfrecords_dir)):
    # Example waveform_2.tfrecord file.split(...)=["waveform","2.tfrecord"]
    if file.endswith(".tfrecord"):
        tfrecords.append(str(os.path.abspath(
            os.path.join(tfrecords_dir, file))))  # The abspath ensures that the string is in path format

with open(path_dict, "rb") as f:
    dictionary_clusters = pickle.load(f)

with open(os.path.join(filter_genres_path, 'genres.txt')) as f:
    genres, merge_list = f.readlines()
    genres = sorted(json.loads(genres))
    merge_list = json.loads(merge_list)

inv_map = {}  # we inverse the dictionary
clusters = np.unique(list(dictionary_clusters.values()))  # clusters is a sorted list
converter = {}
for i in range(len(clusters)):
    converter[clusters[i]] = i

for k, v in dictionary_clusters.items():
    inv_map[converter[v]] = inv_map.get(converter[v], []) + [str(k)]

n_clusters = len(clusters)
merged_genres = []

for l in merge_list:
    if l[0] in genres and l[1] in genres:
        inv_map[l[1]] += inv_map[l[0]]
        merged_genres.append(l)

for l in merged_genres:
    del inv_map[l[0]]
    genres.remove(l[0])


def create_hash(list_tags, main_genre=True):
    ''' Creates the hash table from the clusters dictionary
    
    Parameters
    ----------
    path_dict: str
        Directory containing the dictionary that links the tags with the clusters.
         
    '''

    if main_genre is False:  # The main genre is placed in the first position of the list
        list_tags = list_tags[1:]
    length_encoding = len(list_tags)
    encoded_values = np.eye(length_encoding)
    table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string,
                                                  value_dtype=tf.int64, empty_key="<EMPTY_SENTINEL>",
                                                  deleted_key="<DELETE_SENTINEL>",
                                                  default_value=tf.zeros(length_encoding, dtype=tf.int64))
    table.insert(list_tags, encoded_values)
    return table, length_encoding


@tf.autograph.experimental.do_not_convert
def _parse_features(example, features_dict, shape):
    ''' Parses the serialized tf.Example. '''

    features_dict = tf.io.parse_single_example(example, features_dict)
    features_dict['audio'] = tf.reshape(tf.sparse.to_dense(features_dict['audio']), shape)
    features_dict['tags'] = tf.reshape(tf.sparse.to_dense(features_dict['tags']), (-1,))

    return features_dict


@tf.autograph.experimental.do_not_convert
def _tag_filter(features_dict, hash_table):
    ''' Removes the tids from the dataset that maps into tags without embedding.
    
    Parameters
    ----------
    features_dict: dict
        Dict of features (as provided by .filter).

    hash_table: hash
        Table that links the tags with the encoding of the clusters.

    '''

    tags = features_dict['tags']
    tags = tf.reduce_sum(hash_table.lookup(tags))
    return tf.math.greater(tags, 1)


@tf.autograph.experimental.do_not_convert
def _tag_filter_hotenc_mask(features_dict, hash_table):
    ''' Transforms the string representation of the tags into a hot-encoded vector with the same lenght as the list 'tags'.
    
    Parameters
    ----------
    features: dict
        Dict of features (as provided by .map).
        
    hash_table: hash
        Table that links the tags with the encoding of the clusters.

    mode: string
        Indicates if the model predicts the pdf or the tags

    '''

    tags = features_dict['tags']
    tags = tf.reduce_sum(hash_table.lookup(tags), 0)
    tags = tf.cast(tf.cast(tags, tf.bool), tf.int64)  # transform the components with value greater than one in 1
    return tags


@tf.function
def _tag_to_matrix(tags):
    """Transforms the encoded vector of tags into a single-track co-occurence matrix"""

    example1 = tf.cast(tags, tf.bool)
    example2 = tf.cast(tf.expand_dims(tags, axis=-1), tf.bool)
    mat = tf.cast(tf.logical_and(example1, example2), tf.int32)
    return mat


keys_tensor = tf.constant(list(dictionary_clusters.keys()))

clusters = np.unique(list(dictionary_clusters.values()))  # clusters is a sorted list
length_encoding = len(clusters)
converter = {}

for i in range(len(clusters)):
    converter[clusters[i]] = i

new_values = np.zeros((len(dictionary_clusters), length_encoding), dtype=np.int64)

for i, j in enumerate(dictionary_clusters.values()):
    new_values[i, converter[j]] = 1

table = tf.lookup.experimental.DenseHashTable(key_dtype=tf.string,
                                              value_dtype=tf.int64, empty_key="<EMPTY_SENTINEL>",
                                              deleted_key="<DELETE_SENTINEL>",
                                              default_value=tf.zeros(n_clusters, dtype=tf.int64))
table.insert(keys_tensor, new_values)

AUDIO_FEATURES_DESCRIPTION = {'audio': tf.io.VarLenFeature(tf.float32), 'tid': tf.io.FixedLenFeature((), tf.string),
                              'tags': tf.io.VarLenFeature(tf.string)}  # tags will be added just below

tfrecords = np.array(tfrecords, dtype=np.unicode)  # allow for single str as input
tfrecords = np.vectorize(lambda x: os.path.abspath(os.path.expanduser(x)))(
    tfrecords)  # fix issues with relative paths in input list
datasets = []
dataset = tf.data.TFRecordDataset(tfrecords)
# parse serialized features
dataset = dataset.map(lambda x: _parse_features(x, AUDIO_FEATURES_DESCRIPTION, (-1,)),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

for i in genres:
    print('starting processing for genre ' + str(i))
    with open(os.path.join(args.save_tags_path, 'genre_' + str(i) + '.npy'), 'wb') as f:
        np.save(f, np.array(inv_map[i]))

    dataset_genre = dataset
    hash_table, n_genres = create_hash(inv_map[i])
    dataset_genre = dataset_genre.filter(lambda y: _tag_filter(y, hash_table))
    dataset_genre = dataset_genre.map(lambda y: _tag_filter_hotenc_mask(y, hash_table))
    dataset_genre = dataset_genre.map(lambda y: _tag_to_matrix(y))
    mat = tf.zeros((n_genres, n_genres), dtype=tf.int32)
    mat = dataset_genre.reduce(mat, tf.math.add)
    with open(os.path.join(args.save_co_occurence_matrix_path, 'genre_co-occurrence_' + str(i) + '.npy'), 'wb') as f:
        np.save(f, mat=mat.numpy())
    list_examples = []
    dataset_genre_without_main = dataset
    hash_table, n_genres = create_hash(inv_map[i], False)
    dataset_genre_without_main = dataset_genre_without_main.filter(lambda y: _tag_filter(y, hash_table))
    dataset_genre_without_main = dataset_genre_without_main.map(lambda y: _tag_filter_hotenc_mask(y, hash_table))
    for example in dataset_genre_without_main:
        list_examples.append(example.numpy())
    list_examples = np.vstack(list_examples)
    with open(os.path.join(args.save_tag_vectors_path, 'genre_samples_without_main_' + str(i) + '.npy'), 'wb') as f:
        np.save(f, list_examples)
