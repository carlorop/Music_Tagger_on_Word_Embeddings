""""
Inspection of the confusion matrix
The first axis of the confusion matrix represents: TP TN FP FN
"""

parser = argparse.ArgumentParser()
parser.add_argument("--confusion-matrix-path", dest='confusion_matrix_path',
                    help='full path to the npy file that contains an array representing the confusion matrix')
args = parser.parse_args()

confusion_matrix = np.load(args.confusion_matrix_path)
N = (np.sum(confusion_matrix, axis=0))[0]
confusion_matrix_normalized = confusion_matrix / N

print('Percentage of positive predictions for threshold 0.5: ',
      np.mean(confusion_matrix_normalized[:, 100, :], axis=1)[0] +
      np.mean(confusion_matrix_normalized[:, 100, :], axis=1)[2])

index = 24  # Corresponds to "rock" in the model with optimum hyper-parameters
thresholds = [0.33, 0.5, 0.67]
for q in thresholds:
    print('Using threshold ', q, ' the confussion matrix is',
          WE_basemodel_confussion_matrix_normalized[:, int(200 * q), index_WE])
