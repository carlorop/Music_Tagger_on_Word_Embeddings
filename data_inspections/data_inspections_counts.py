"""
Contains different data inspections, mainly used to count the number of tags or clusters in different models 
"""

import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from lastfm import LastFm

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", dest='model_path',
                    help='full path of the pickle file that contains the dictionary of clusters')
parser.add_argument('--filter-path', dest='filter_path',
                    help='directory that contains tags_remove.txt, this file contains the name of the tags that we want to remove, the format of the file is: "tag_1 \n tag_2 \n..."')
parser.add_argument("--last-fm", dest='last_fm',
                    help='full path of the dataset of lastfm tags, the format of the dataset must be .db ',
                    default="lastfm_tags.db")
args = parser.parse_args()

fm = LastFm(args.last_fm)
tags = fm.popularity()
total_number_counts = np.sum(tags['count'].tolist())
total_number_tags = len(tags)
tags_100 = list(fm.popularity()['tag'][:100])
tags_100_original = tags_100.copy()
tags_100_counts = np.sum(tags['count'].loc[tags['tag'].isin(tags_100_original)].tolist())
print('Total number of counts: ', total_number_counts)
print('Number of counts in the 100 most popular tags: ', tags_100_counts)

if args.filter_path:
    path = args.filter_path
    file = 'tags_remove.txt'
    filtered_tags_number_counts = np.sum(tags['count'].loc[tags['tag'].isin(tags_100)].tolist())

    with open(os.path.join(path, file), 'r') as f:
        tags_remove = f.read().split('\n')
        for i in tags_remove:
            if i in tags_100:
                tags_100.remove(i)

    print('Counts of the filtered tags: ', filtered_tags_number_counts,
          ', pectentage with respect to the total number of counts: '
          , 100 * filtered_tags_number_counts / total_number_counts, ' %')

if args.model_path:
    with open(args.model_path, "rb") as f:
        dictionary_clusters = pickle.load(f)
    tags_cluster = list(dictionary_clusters.keys())
    print('number of clusters:', len(np.unique(np.array(list(dictionary_clusters.values())))))
    clusters_number_counts = np.sum(tags['count'].loc[tags['tag'].isin(tags_cluster)].tolist())
    clusters_number_tags = len(tags['tag'].loc[tags['tag'].isin(tags_cluster)])
    print('Number of counts in the set of clusters: ', clusters_number_counts,
          ', pectentage with respect to the total number of counts: ',
          100 * clusters_number_counts / total_number_counts, ' %')
    print('The clustering removes ', 1 - (clusters_number_tags) / total_number_tags, '% of tags')
    inv_map = {}  # we compute the inverse dictionary
    for k, v in dictionary_clusters.items():
        inv_map[v] = inv_map.get(v, []) + [str(k)]

    counts_map = {}
    for i in inv_map.keys():
        counts_map[i] = np.sum(tags['count'].loc[tags['tag'].isin(inv_map[i])].tolist())
        print('cluster ', str(i), ' has ', str(len(inv_map[i])), ' words and ', counts_map[i], ' counts')

    plt.hist(list(counts_map.values()), histtype='bar', ec='black')

    plt.title('Counts per cluster')
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.show()
