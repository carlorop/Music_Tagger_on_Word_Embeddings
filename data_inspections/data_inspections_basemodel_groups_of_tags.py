""""
Contains data inspections related to the results of the baseline models
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--confusion-matrix-path-filtered", dest='confusion_matrix_path_filtered',
                    help='full path to the npy file that contains an array representing the confusion matrix for the baseline filtered model')
parser.add_argument("--confusion-matrix-path-baseline", dest='confusion_matrix_path_baseline',
                    help='full path to the npy file that contains an array representing the confusion matrix for the baseline model')
parser.add_argument("--ROC-per-label-path-filtered", dest='ROC_per_label_path_filtered',
                    help='full path to the npy file that contains an array representing the AUC ROC per label for the baseline filtered model')
parser.add_argument("--PR-per-label-path-filtered", dest='PR_per_label_path_filtered',
                    help='full path to the npy file that contains an array representing the AUC PR per label for the baseline filtered model')
parser.add_argument("--ROC-per-label-path-baseline", dest='ROC_per_label_path_baseline',
                    help='full path to the npy file that contains an array representing the AUC ROC per label for the baseline model')
parser.add_argument("--PR-per-label-path-baseline", dest='PR_per_label_path_baseline',
                    help='full path to the npy file that contains an array representing the AUC PR per label for the baseline model')
parser.add_argument("--list-tags-baseline", dest='list_tags_baseline',
                    help='full path to the npy file that contains an array containing the names of the tags for the baseline model')
args = parser.parse_args()

"""
Filtered model
"""

basemodel_filtered_confusion_matrix = np.load(args.confusion_matrix_path_filtered)
PR_filtered = np.load(args.PR_per_label_path_filtered)
ROC_filtered = np.load(args.ROC_per_label_path_filtered)
FP_half_filtered = basemodel_filtered_confusion_matrix[0, 100, :]

plt.hist(PR_filtered[-1, :], histtype='bar', ec='black', density=True)
plt.title('Normalized histogram of AUC PR - filtered basemodel')
kde = stats.gaussian_kde(PR_filtered[-1, :])
xx = np.linspace(0, 0.8, 1000)
plt.plot(xx, kde(xx))
plt.xlim([0.0, 0.7])
plt.ylim([0.0, 4.45])

plt.xlabel('AUC PR')
plt.legend(['KDE', 'normalized counts'])
plt.show()

# The ordering of the tags may vary between different runs of the training script. This ordering correspond to the ROC file contained in https://github.com/carlorop/master_project.git
tags_filtered = np.array(['male vocalist', 'death metal', 'chillout', 'hip hop',
                          'heavy metal', 'melancholy', 'reggae', 'beautiful', 'ambient',
                          'funk', 'oldies', 'summer', 'rock', 'punk rock', 'emo',
                          'electronica', 'female vocalist', 'indie pop', 'electro', 'female',
                          'party', 'hard rock', 'guitar', 'sad', 'pop', '70s', 'electronic',
                          'fun', 'instrumental', 'pop rock', 'alternative rock',
                          'psychedelic', 'downtempo', 'trance', 'metal', 'upbeat',
                          'relaxing', '90s', 'blues', 'Hip-Hop', 'hardcore', 'soul',
                          'Progressive rock', 'Love', '80s', 'singer-songwriter', 'country',
                          'rnb', 'lounge', 'dance', 'male vocalists', 'punk',
                          'easy listening', 'rap', '2000s', '00s', 'female vocalists',
                          'new wave', 'cover', 'romantic', 'alternative', 'techno', 'Mellow',
                          'piano', 'melancholic', 'folk', 'indie rock', 'jazz', 'indie',
                          'classic rock', 'relax', 'chill', 'experimental', 'House', '60s',
                          'acoustic', 'happy', 'Ballad', 'loved', 'classic'])

genre_filtered = [1, 2, 3, 4, 6, 9, 12, 13, 14, 15, 17, 18, 21, 24, 26, 29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 46,
                  47, 49, 51, 53, 57, 60, 61, 65, 66, 67, 68, 69, 72, 73, 77, 79]
genre_names = tags_filtered[
    [1, 2, 3, 4, 6, 9, 12, 13, 14, 15, 17, 18, 21, 24, 26, 29, 30, 31, 32, 33, 34, 38, 39, 40, 41, 42, 46, 47, 49, 51,
     53, 57, 60, 61, 65, 66, 67, 68, 69, 72, 73, 77, 79]]

Instruments_filtered = [22, 63]
Instruments_names = tags_filtered[[22, 63]]

Dates_filtered = [10, 25, 37, 44, 54, 55, 74]
Dates_names = tags_filtered[Dates_filtered]

Voices_filtered = [0, 16, 19, 50, 56]
Voices_names = tags_filtered[Voices_filtered]

Emotions_filtered = [5, 7, 20, 23, 27, 35, 36, 43, 52, 59, 62, 64, 70, 71, 76, 78]
Emotions_names = tags_filtered[Emotions_filtered]

Miscellaneous_filtered = [8, 11, 28, 45, 48, 58, 75]
Miscellaneous_names = tags_filtered[[8, 11, 28, 45, 48, 58, 75]]

Nationalities_names = ['UK', 'USA', 'british', 'american']

print('genres', np.mean(PR_filtered[-1, genre_filtered]))
print('Instruments', np.mean(PR_filtered[-1, Instruments_filtered]))
print('Dates', np.mean(PR_filtered[-1, Dates_filtered]))
print('Voices', np.mean(PR_filtered[-1, Voices_filtered]))
print('Emotions', np.mean(PR_filtered[-1, Emotions_filtered]))
print('Miscellaneous', np.mean(PR_filtered[-1, Miscellaneous_filtered]))

FP_half_filtered_genre = FP_half_filtered[genre_filtered]

"""
Baseline model
"""

basemodel_filtered_confusion_matrix = np.load(args.confusion_matrix_path_filtered)
PR_filtered = np.load(args.PR_per_label_path_filtered)
ROC_filtered = np.load(args.ROC_per_label_path_filtered)
basemodel_confusion_matrix = np.load(args.confusion_matrix_path_baseline)
basemodel_tags = np.load(args.list_tags_baseline)
PR = np.load(args.PR_per_label_path_baseline)
ROC = np.load(args.ROC_per_label_path_baseline)

plt.hist(PR[-1, :], histtype='bar', ec='black', density=True)
plt.title('Normalized histogram of AUC PR - basemodel')
kde = stats.gaussian_kde(PR[-1, :])
xx = np.linspace(0, 0.8, 1000)
plt.plot(xx, kde(xx))
plt.xlim([0.0, 0.7])
plt.xlabel('AUC PR')
plt.legend(['KDE', 'normalized counts'])
plt.show()

genre = []

Instruments = []

Dates = []

Voices = []

Emotions = []

Miscellaneous = []

Nationalities = []

Subjective = []

for i in range(100):
    if basemodel_tags[i] in genre_names:
        genre.append(i)
    elif basemodel_tags[i] in Instruments_names:
        Instruments.append(i)
    elif basemodel_tags[i] in Emotions_names:
        Emotions.append(i)
    elif basemodel_tags[i] in Dates_names:
        Dates.append(i)
    elif basemodel_tags[i] in Voices_names:
        Voices.append(i)
    elif basemodel_tags[i] in Miscellaneous_names:
        Miscellaneous.append(i)
    elif basemodel_tags[i] in Nationalities_names:
        Nationalities.append(i)
    else:
        Subjective.append(i)

print('Baseline model:')
print('genres', np.mean(PR[-1, genre]))
print('Instruments', np.mean(PR[-1, Instruments]))
print('Dates', np.mean(PR[-1, Dates]))
print('Voices', np.mean(PR[-1, Voices]))
print('Emotions', np.mean(PR[-1, Emotions]))
print('Miscellaneous', np.mean(PR[-1, Miscellaneous]))
print('Nationalities', np.mean(PR[-1, Nationalities]))
print('Subjective', np.mean(PR[-1, Subjective]))

FP_half = basemodel_confusion_matrix[0, 100, :]
FP_half_genre = FP_half[genre]
