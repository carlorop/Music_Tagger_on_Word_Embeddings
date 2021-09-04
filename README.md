# Incorporating Word Embeddings in Semantic Label Deep Learning Classifiers


## Introduction

This is the repository of an Imperial College master's project related with deep neural networks for music classification. The project uses the CNN presented by J. Pons et al. in [this](https://arxiv.org/pdf/1711.02520.pdf) paper as starting point. The dataset is a combination of the [Million Song Dataset](http://millionsongdataset.com/) and the  [last.fm tags](http://millionsongdataset.com/lastfm/). The latter dataset has been created using crowdsourcing techniques. These kind of datasets usually contain a large number of labeling errors, for this reason, only the most populartags are used in practice, losing a substantial amount of information from the dataset.  The main purpose of this project is to introduce techniques to maximize the retrieval of information fromdatasets of labeled music without compromising the quality of the information. These methods shall be based on the concept of word embeddings.

The python version is 3.6. The preprocessing of the audio and datasets is performed over the audio samples provided by 7digital. The files used for preprocessing and some other scripts have been taken from [this](https://github.com/pukkapies/urop2019) repository. We use `preprocessing_all_tags.py` to create TFRecord files that encode the audio of the sample, the track ID (TID) and the list of tags, stored as a list of strings. The inputs of this script are the aforementioned datasets and the file `ultimate.csv`, which contains information about the correspondence between the elements of the datasets and has been taken from the foregoing [repository](https://github.com/pukkapies/urop2019). 

We shall explain the usages of the scripts of the repository in the explaination of the models.


## Models on tags

### Model 1: Classification on raw tags


We may consider this model as an starting point, we will train the wave-front and spectrogram classifiers on the tags without filtering, for this purpose we will just consider the 100 most popular tags of the last.fm dataset, discarding the rest. We chose such a high number, compared with the common choice of 50 tags, because we will reduce gradually the number of tags using different filtering mechanisms along our models. This subset of 100 tags encompasses 1,929,985 counts, which corresponds to 22.4452% of the counts of the original dataset. This model does not aim to be a comparison with the results of J. Pons et al., since the number of tags is different and their method for averaging the metrics (micro averaged or macro averaged) is unknown.

We shall measure the performance of the models using both the macro-averaging and the micro-averaging scheme for assessing the models. We recall that in the macro-averaging scheme mode the metrics are computed for each label and the result correspond to the mean of these metrics, in the micro-averaging the metrics are computed from a unique confusion matrix which is the sum of the confusion matrices for each labels. Since the dataset is largely imbalanced, the less frequent tags will almost not affect to the performance in the micro-averaging scheme, some tags are subjective and therefore they will lead to a poor within-label metric, reducing the macro-averaging measure independently of the goodness of the model.


### Model 2: Classification on filtered tags
As we have seen in the previous model, the tags with the lowest performance either are based on features that are unrecognizable for the classifier or are completely subjective. Therefore, in this model we train the classifier on a set of filtered labels. We will remove the tags showed in the following table:


| favorites      | Awesome | sexy     | catchy     | Soundtrack       |
|----------------|---------|----------|------------|------------------|
| USA            | UK      | british  | american   | heard on Pandora |
| seen live      | cool    | Favorite | Favourites | favourite        |
| favorite songs | amazing | good     | epic       | Favourite Songs  |

These labels correspond to highly subjective tags which cannot be recognised from musical features. See that we have not removed the tags associated with emotions since, even when these tags are subjective, they are related with musical features. For instance, the tag "happy" could arise in songs with fast tempo.


### Usage of scripts

The scripts used to run the models are found in `main_models/classifiers_on_tags/`. You need to specify the configurations in a json file named `config.json`. The models are trained in the script `train.py`. This script loads the configuration from `config.json` using functions from `orpheus_model.py`, furthermore, it processes the TFRecords and splits the samples into training and validation set using functions from `data_input.py`. 

The parameters of the training script are:
| Parameter                 | Description   |	
| :------------------------ | :-------------|
| frontend 	       |Specify the format of the input, 'waveform' or 'log-mel-spectrogram'
| --tfrecords-dir 	       |directory to read the .tfrecord files 
| --config | path to config.json 
| --lastfm | path to lastfm database 
| --multi-db | specify the number of different tags features in the .tfrecord files
| --filter', action='store_true | Boolean for filtering the tags 
| --filter-path | directory that contains tags_remove.txt, this file contains the name of the tags that we want to remove, the format of the file is: "tag_1 \n tag_2 \n..." 
| --save-PR | path in which we save a npy file containing an array representing the AUC PR per label if multi-tags is True 
| --save-ROC | path in which we save a npy file containing an array representing the AUC ROC per label if multi-tags is True 
| --save-confusion-matrix | path in which we save a npy file containing an array representing the confusion matrix per label if multi-tags is True 
| --save-tags-path | path to save an array containing the filtered tags for interpreting each component of the classification 
| --multi-tag | If true, we compute the macro-averaged metrics, otherwise, we compute the micro-averaged ones 
| --multi-db-default | specify the index of the default tags database, when there are more than one tags features in the .tfrecord files
| --epochs | specify the number of epochs to train on
| --steps-per-epoch | specify the number of steps to perform for each epoch (if unspecified, go through the whole dataset 
| --no-shuffle | force no shuffle, override config setting 
| --update-freq | specify the frequency (in steps) to record metrics and losses
| --cuda | set cuda visible devices
| --built-in | train using the built-in model.fit training loop 
| -v –verbose | verbose mode, options=['0', '1', '2', '3'] 



*Example:*

```bash
# Model 1
python train.py waveform --epochs 30 --tfrecords-dir  /srv/data/tfrecords/waveform-complete  --config  /home/carlos/configs/Wconfig --lastfm /home/carlos/srv/data/urop/lastfm_tags.db --cuda 2 
```

```bash
# Model 2
python train.py waveform --epochs 30 --tfrecords-dir  /srv/data/tfrecords/waveform-complete  --config  /home/carlos/configs/Wconfig --lastfm /home/carlos/srv/data/urop/lastfm_tags.db --filter --filter-path /home/carlos --save-PR /home/carlos/saved_models/figures --save-ROC /home/carlos/saved_models/figures --save-confusion-matrix /home/carlos/saved_models/figures  --cuda 2 
```

The script `custom_metric.py` contains a modification of the `tf.keras.metrics.AUC` which returns the per-label AUC PR and AUC ROC. The file `lastfm.py` contains methods for analysing the last.fm dataset.


## Models on clusters

The main motivation of the project is maximizing the information retrieval from the dataset. The last.fm dataset have major drawbacks. One of them is that some tags are highly subjective, other common issue is the large number of labels, which is impractical for classification, furthermore, there may exist several different labels that refer to the same concept. For this reason, in the last.fm dataset the classification is usually performed on a small subset of the labels, therefore, most of the tags of the dataset are discarded, losing a large amount of relevant information about the samples. In this model, we will propose a method to retrieve information from the less popular tags. This model consists in classifying the samples in classes that contain several similar tags. It is based on clusterings of word embeddings. In this way, we may create labels that encompass tags with similar meanings, such as "Allegro" and "Fast paced" or genres and their subgenres.

First, we assign a word embedding to each tag in the dataset. See that the smaller the angle between embeddings is, the more similar the tags are. On the other hand, the semantic information does not depend on the length of the embeddings, for convenience, we will normalize the word embeddings before clustering. Let us define the similarity measure d(x,y) as d(x,y)=1-cos(x,y), where cos(x,y) is the angle between the word embeddings x and y. If the word embeddings are normalized they verify ||x-y||<sup>2</sup>=2d(x,y). Hence, clustering according to teh euclidean distance will be equivalent to cluster according d(x,y), which is a measure of the similarity between words.

Since the dataset is large, we will use a slight variation of the k-means clustering algorithm normalizing the centroids. This algorithm can be found in `embeddings/k_means_normalized.py`. Using the variation of the k-means, we will group the tags into k clusters, which will be the classes of our classification problem. We would expect that, from the semantic information carried by the tags, each cluster captures at least a feature that can be detected in the tracks. In this way, we may increase the amount of information retrieved from the dataset, since we will not classify into tags but into their meanings, and therefore we will also retrieve information from tags with fewer counts that otherwise we would have rejected.

The clusters may be created in `embeddings/compute_dictionaries_k_means.py`, specifying the hyper-parameters: number of clusters, embedding algorithm and the occurence threshold. The dictionaries are saved as a pickle file. We have stored the dictionaries for all the models in `dictionary_clusters/`. The script `embeddings/embeddings_projector.py` creates files to project the word embeddings into the tensorflow embedding [projector](https://projector.tensorflow.org/).


### Model 3: Model on clusters

In this model we trained a classifier on a set of clustered labels according to the similarity of the word embeddings of the tags. We have included in the clustering only the tags whose number of counts is greater than a given threshold. the dictionary linking each tag with its cluster is stored in `dictionary_clusters/embeddings_baseline`. This clustering led a decrease of the performance of the classifier due to the lack of filtering. Therefore, in the following models we removed all the clusters that do not contain any filtered tag used in Model 2. We have performed a grid search on the following hyper-parameters: Dataset of embeddings, threshold of counts and number of clusters before filtering. The datasets of embedding used were the following: Word2Vec algorithm over the English Wikipedia [corpus](https://tfhub.dev/google/Wiki-words-250-with-normalization/2), the corresponding dictionaries are stored in `dictionary_clusters/word2vec`;  GloVe algorithm trained on the English Wikipedia [corpus](https://nlp.stanford.edu/data/glove.6B.zip), stored in `dictionary_clusters/glove-wikipedia`; and the GloVe model trained on the twitter [corpus](https://nlp.stanford.edu/data/glove.twitter.27B.zip), in `dictionary_clusters/glove-twitter`. The directories `100_models` and `500_models` contain the dictionaries generated using 100 and 500 counts respectively. The name of the dictionaries follows the template `tags_dict_x.pkl`, where x represents the number of clusters before clustering. The clustering that led to the best performance was the one generated by the Glove model trained on the Wikipedia corpus with threshold 100 and 100 clusters before filtering.

We have also trained the classifier using just clusters of genres. The base clusters are the ones generated by the optimum set of hyper-parameters. File `genres.txt` contains a list of clusters of genres and a list with pairs of clusters that are higly correlated, i.e., cluster with a high number of co-occurwences in the tracks.

### Usage of scripts

The scripts used to run the models are found in `main_models/classifiers_on_clusters/`. The configuration of the model is specified in `config/config_clusters/config.json`. The model is trained using `train.py`, whose dependencies are analogous to the previous model.

The parameters of the training script are the following, the required parameters are marked in bold:
| Parameter                 | Description   |	
| :------------------------ | :-------------|
|** frontend **	       |Specify the format of the input, 'waveform' or 'log-mel-spectrogram'
|**--tfrecords-dir** 	       |directory to read the .tfrecord files 
|**--dictionary-dir** | directory to read the dictionary that links the words and the clusters
|**--config**  | path to config.json 
| --multi-db | specify the number of different tags features in the .tfrecord files
| --filter', action='store_true | Boolean for filtering the tags 
| --filter-path | directory that contains tags_remove.txt, this file contains the name of the tags that we want to remove, the format of the file is: "tag_1 \n tag_2 \n..." 
| --save-PR | path in which we save a npy file containing an array representing the AUC PR per label if multi-tags is True 
| --save-ROC | path in which we save a npy file containing an array representing the AUC ROC per label if multi-tags is True 
| --save-confusion-matrix | path in which we save a npy file containing an array representing the confusion matrix per label if multi-tags is True 
| --multi-tag | If true, we compute the macro-averaged metrics, otherwise, we compute the micro-averaged ones 
| --multi-db-default | specify the index of the default tags database, when there are more than one tags features in the .tfrecord files
| --epochs | specify the number of epochs to train on
| --steps-per-epoch | specify the number of steps to perform for each epoch (if unspecified, go through the whole dataset 
| --no-shuffle | force no shuffle, override config setting 
| --update-freq | specify the frequency (in steps) to record metrics and losses
| --cuda | set cuda visible devices
| --built-in | train using the built-in model.fit training loop 
| -v –verbose | verbose mode, options=['0', '1', '2', '3'] 
| --built-in | train using the built-in model.fit training loop 
|--save-coocurrence-matrix-path | path to the directory in which we save the co-occuernce matrix


*Example:*

```bash
# Model 3
python train.py log-mel-spectrogram --epochs 20 --tfrecords-dir /srv/data/tfrecords/log-mel-complete --config /home/carlos/configs/WEconfig  --dictionary-dir /home/carlos/models/comparison-genres  --save-PR /home/carlos/saved_models/figures --save-ROC /home/carlos/saved_models/figures --save-confusion-matrix /home/carlos/saved_models/figures  --multi-tag --cuda 0 1 2 3 
```
To train just on genres, we must pass also the following parameter:

| Parameter                 | Description   |	
| :------------------------ | :-------------|
|--genre-path | path to the directory which containts genres.txt, a file containing the clusters with genres and a list of highly correlated clusters, the txt is composed by two lines, the first one is a list of clusters [x,..,z] and the second one is a list of pairs of clusters to be merged [[x,y],...]')





## Model 4: Models on clusters based on co-occurrence
The clusters of genres contain several subgenres of a main genre, which is usually the most popular tag in the cluster. The labels of the following models will be clusters of subgenres. The number of counts of the subgenres of any genre is significantly less than the total counts, therefore, we will be able to create clusters based on co-occurrence. We shall create clusters of tags that belong to the same genre, classifying on the union these clusters.

See that the scripts used to train the following models are the same as in Model 3. The difference is found in the method used to cluster the labels. 
### Model 4.1: Classification on subgenres via NPMI
In this model, we will consider the tags as elements of a discrete space, hence, we will use the agglomerative hierarchical algorithm to perform the clustering. To that end, we are going to define a measure of similarity that captures the co-occurrences between the elements of the genre cluster. We shall define this similarity from the concept of pointwise mutual information (PMI). The PMI is a measure of association commonly use in NLP which is derived from the co-occurrence matrix, it has been successfully to problems in the field of NLP, such as sentiment analysis or review classification. We may use the PMI to define a similarity measure between tags `d(i,j)` which verifies `d(i,i)=0` and  `d(i,j)=d(j,i)` as:


<img src="https://render.githubusercontent.com/render/math?math=d(i,j)%20=1-N%20P%20M%20I\left(i,j\right)=%20\frac{P%20M%20I\left(i,j\right)}{-\log%20P(i,%20j)}=\frac{\log%20(P(i)P(j))}{\log%20P(i,j)}-1">

We have clustered the tags of each of the clusters of genres. We have used an agglomerative approach with `d(i,j)` as similarity function. We have computed the clustering using from `5` up to `20` clusters, selecting the set of clusters that maximize the silhouette score and removing all the singletons. We have performed the embeddings in `embeddings/skip-gram.py`.  We have compared different linkage criteria, selecting the average linkage since it led to the best results. The resulting dictionary is stored in `dictionary_clusters/npmi`

### Model 4.2: Classification on subgenres via tag embeddings

In this model, we will create embeddings of the classes of the classification problem. If there are N classes in the problem, we may associate class i with an N-dimensional vector whose components are 0 except for a 1 in position i. As in the one-hot encoding of the words of a vocabulary, such vectors do not contain any information about the relations of the classes. We may associate a tag embedding to each class, with dimension d<N, that captures the relations between different labels.

We shall associate the tag embeddings to tags that belong to the same cluster of genres. We will replicate the skip-gram model presented by T. Mikolov et al. in [this]( https://arxiv.org/abs/1301.3781) paper, adapting it to our problem. The neural network will consist in three layers: the input layer, whose dimension is the size of the vocabulary, in our case, the number of tags in the cluster (N); the hidden layer, which transforms the input with size N in an output with dimension d; and the output layer, whose output has size N. The word embedding of tag i would correspond to the i-th row of the weight matrix associated to the hidden layer.

The input will be a one-hot encoding representing the tag, since the number of tags that belong to the same genre per track is reduced, we will not impose a window size, we will try to predict all the tags from the tracks but the input tag. For instance, if the combination of tags of a track is represented as [0,1,1,1], we would pass this vector three times to the neural network, it would correspond to the following samples ([0,1,0,0],[0,0,1,1]), ([0,0,1,0],[0,1,0,1]) and ([0,0,0,1],[0,1,1,0]), where the first element is the input and the second one is the ground truth. 
As a rule of thumb, the dimension of the word embeddings is usually the fourth root of the dimension of the vocabulary. In our case the vocabulary is the set of tags that belong to a cluster, whose order of magnitude is approximately 10 <sup>2</sup>, hence, we will take the floor function of the square root of N as the dimension of the embeddings, instead of the fourth root.

We have trained the neural network using the binary cross entropy as loss function and Adam as optimizer. We have performed the embeddings in `embeddings/skip-gram.py`. The resulting word embeddings have been normalized and clustered using k-means algorithm, the resulting dictionary is stored at `dictionary_clusters/tag_embeddings`

## Model on tags with custom loss function
### Model 5: Custom loss function

In this model we will explore a different way of employing word embeddings to take advantage of the semantic information of the tags. We will classify on the 80 filtered tags from Model 2. However, we shall modify the loss function to penalise predictions of tags whose semantic meaning is inconsistent with the true labels of the track. The loss function that we have used in the previous models is the binary cross-entropy, nonetheless, in this model we shall use the following loss function per sample:

<img src="https://render.githubusercontent.com/render/math?math=\mathcal{L}\left(\theta,\mathcal{M}\right)\left(\hat{y}_{i}\right)=-\frac{1}{n}\sum_{i=1}^{n}\left(t_{i} \log\left(\hat{y}_{i}\right)+\left(1-t_{i}\right)\log \left(1-\hat{y}_{i}\right)(1+\gamma (1-\max_{j\in\mathcal{T}}(\omega_i\cdot\omega_j)))\right)">

Where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}"> represents the set of true labels of the sample, <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_{i}"> is the probability prediction for label i, <img src="https://render.githubusercontent.com/render/math?math=\gamma"> is a hyper-parameter, <img src="https://render.githubusercontent.com/render/math?math=\omega_i"> represents a normalized word embedding and <img src="https://render.githubusercontent.com/render/math?math=t_i"> equals 1 if i is one of the true labels of the sample and 0 otherwise. See that <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_{i}\in (0,1)">, hence, for each iteration of the sum in which <img src="https://render.githubusercontent.com/render/math?math=t_i">, the modification of the loss function is equivalent to adding a positive term proportional to <img src="https://render.githubusercontent.com/render/math?math=(1-\max\limits_{j\in \mathcal{T}}(\omega_i\cdot \omega_j))">, to the binary cross entropy. The justification of the max operator can be understood easily with an example. Let us assume that we pass a sample with the tags "rock" and "female voices". We expect a large value for <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_{i}">, in the case of the tag "female voice", which lead to a large value of <img src="https://render.githubusercontent.com/render/math?math=\log(1-\hat{y}_{i})">, in this case we would induce a small penalisation by adding the term <img src="https://render.githubusercontent.com/render/math?math=\log(1-\hat{y}_{i})\gamma(1-\max\limits_{j\in \mathcal{T}}(\omega_i\cdot \omega_j))">, since the cosine similarity between the embeddings of "female voice" and "female voices" is close to 1 independently of the similarity between "female voice" and "rock". See that if we had averaged the cosine similarity it would have led to random weights.

### Usage of scripts

This model is trained using the same scripts as in Model 1 and Model 2. With the addition of the following hyper-parameters:

| Parameter                 | Description   |	
| :------------------------ | :-------------|
| --gamma | Hyperparameter used in the custom loss function
| --custom-loss-embeddings-path | If exist, we use the custom loss function. Full path to txt file encoding the glove pre-trained embeddings, these embeddings can be downloaded from https://nlp.stanford.edu/projects/glove/ 



