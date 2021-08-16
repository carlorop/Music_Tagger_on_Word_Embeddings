"""
In this file we create the files needed for projecting the word embeddings into the tensorflow embedding projector https://projector.tensorflow.org/
"""
import argparse
import os

import tensorflow_hub as hub
from lastfm import LastFm

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", dest='threshold',
                    help='specify the threshold, we remove all teh tags whose number of counts in the last.fm dataset is less than the threshold',
                    type=int, default=100)
parser.add_argument("--save-path", dest='save_path', help='directory in which we save the files')
parser.add_argument("--lastfm", help='full path to the lastfm tags database')

args = parser.parse_args()

embed = hub.load("https://tfhub.dev/google/Wiki-words-250-with-normalization/2")

n = args.threshold
fm = LastFm(args.lastfm)
tags = fm.popularity()
tags = tags[tags['count'] > n]['tag'].tolist()
path = os.path.join(args.save_path, 'metadata.tsv')

failed_tags = []
with open(path, "w") as f:
    for i, tag in enumerate(tags):
        try:
            f.write("{}\n".format(tag))
        except:
            failed_tags.append(i)

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

path = os.path.join(args.save_path, 'vectors.tsv')
with open(path, "w") as f:
    for tag in tags_embeddings:
        string = str(list(tag)).replace("[", "").replace("]", "\n").replace(",", "\t").split(" ")
        for comp in string:
            f.write("{}".format(comp))
