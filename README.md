# Amazon_food_review_sentiment_analysis
Dataset used: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

I try to predict whether the review is Positive, Negative or Neutral in nature. The dataset has over 500,000 reviews, out of which I randomly sampled 5000 reviews for my work. The dataset includes reviews submitted by Amazon users for a variety of food products from 1999 to 2012, including snacks, beverages, and cooking supplies.I selected this dataset because it is fairly easy to understand and has robust data records. The fields are easy to interpret and the dataset is described in detail on kaggle. 

Labels and input text

For each review, we have product ID, user ID, rating, review summary, review text, and other attributes.

The dataset also includes a "helpfulness" rating for each review, indicating the degree to which other users thought the review to be helpful. This indicator is determined by comparing the number of users who found it useful to the number of users who voted on the review.
When I preprocessed the data, I removed duplicates, and dropped all columns which were not necessary for the task, like: 

'Id','UserId','ProfileName','Time'

After this, I concatenated the summary field to the text field, so that I can analyse the summary along with the text, as summary is equally important and can be quite helpful.
Ultimately, I sent product ID and Text (with summary concatenated) as input fields, and Score as output label (based on which we find out if the review was positive, negative or neutral).

Training, Validation and Test Split

After preprocessing the data, I split the data x (with all input fields), and y (with output field) into 3 parts: Training (60%), validation (20%) and testing (20%).
I did this using train_test_split from sklearn.model_selection. 
After performing this split on 5000 rows of my data, I obtained the following split:



This table shows the number of records in each split, specified by input and output fields as well.

Clustering

As stated in the coursework specifications, in this section, I applied my own implementation of KMeans clustering over my dataset. 
To implement this, I followed the following steps:

Vectorization of the text data: For this step, I wrote a pipeline for tokenizing the text data passed as parameters. I performed tokenization, lemmatization, and stop-word removal on a given text string with the help of spaCy. I also used lxml in order to remove all the html tags that were returned along with the tokens because they wouldn’t be helpful for performing text classification.

Pick k random "centroids": I wrote a function (initialCentroids(k)) which would return a random centroid from the samples as an array. ‘k’ is the number of clusters.

Assign each vector to its closest centroid: By calculating the distances between each data point in the dataset and the current set of centroids, I assign each data point to the nearest centroid based on those distances in the function assign(c).

 Recalculate the centroids based on the closest vectors: The function recalculate_centroids(labels, k) updates the centroid values based on the K-means algorithm's current clustering of the data points. The new set of centroids that represent each cluster's centre are obtained by calculating the mean of all the data points in each cluster.

Finally, I wrote the function kmeans(k=5, max_iterations=100) which uses the above functions as steps in order to perform KMeans clustering. By updating the cluster assignments of the data points and centroids repeatedly until convergence, this function performs the K-means clustering.
After analyzing the clusters, I noticed that:

Cluster 1: It contains reviews related to pet food and treats. All of the example reviews are positive.

Cluster 2: It contains positive reviews about healthy snacks.

Cluster 3: It contains reviews about chips, and about medicinal things.

Cluster 4: It contains reviews related to tea products.It has reviews about different kinds of teas.

Cluster 5: It contains positive reviews, and negative ones  majorly about sweet food.

To conclude, yes the clusters do make sense. We can clearly see that documents assigned to each cluster have similarities to some extent, meaning that the kMeans algorithm works efficiently on our dataset. 
When initially executing the model, I obtained some clusters that had multiple labels, so I noticed that even though they had multiple labels, they were similar. For example: A cluster can have both positive and negative reviews about some kind of tea, etc.

Although the topic of pet food and treats doesn’t appear in other clusters so far, yet the other 4 clusters are almost similar.

Comparing classifiers

I  conducted experiments using the following combinations of classifier models and feature representations:

●Dummy Classifier with strategy="most_frequent"

● Dummy Classifier with strategy="stratified"

● LogisticRegression with One-hot vectorization

● LogisticRegression with TF-IDF vectorization (default settings)

● SVC Classifier with One-hot vectorization (SVM with RBF kernel, default settings)

Along with these baseline classifiers, I used DecisionTreeClassifier from sklearn.tree as my own added classifier to the text data. The decision tree classifier creates a tree-like structure in which each node represents a decision based on the value of a feature by recursively splitting the data into subsets based on the value of a single feature at a time. At each node, the algorithm selects the feature that best separates the data based on some criterion (such as information gain or Gini impurity). The root of the tree represents the entire dataset. Until the algorithm reaches a stopping criterion, such as a maximum depth, a minimum number of samples per leaf node, or a minimum information gain, the decision tree continues to divide the data into smaller subsets.

Parameter Tuning

In this section, I try out different parameters on classifiers and vectorizers to find the optimal parameters for the same.
Classifier - Regularisation C value (typical values might be powers of 10 (from 10^-3 to 10^5)
Vectorizer - Parameters: sublinear_tf (either True or False), max_features (vocabulary size) (in a range None to 50k), and norm (either l1 or l2)
The parameters that I used are as follows:

'C': [0.01, 0.1, 1, 10, 1000, 100000]
When I executed the classifier using the parameters ‘C’ as above, the best F1 score was provided by ‘C’: 100000. So, after this, I used ‘C’: 100000 and checked for the best parameter for ‘max_feature’.

'max_features': [None, 5000, 15000, 25000, 50000]
When I executed the vectorizer using the parameters ‘max_features’ as above, the best F1 score was provided by 'max_features': None. So, after this, I used ‘C’: 100000 for classifier and 'max_features': None for vectorizer  and checked for the best parameter for 'sublinear_tf'.

'sublinear_tf': [True, False]
When I executed the vectorizer using the parameters ‘sublinear_tf'’ as above, the best F1 score was provided by 'sublinear_tf': False,  'max_features': None,  and ‘C’: 100000. So, after this, I used  ‘C’: 100000 for classifier,  'max_features': None and 'sublinear_tf': False  for vectorizer  and checked for the best parameter for 'norm'.


'norm': ['l1','l2']
When I executed the vectorizer using the parameters ‘norm’ as above, the best F1 score was provided by 'sublinear_tf': True,  'max_features': None, ‘norm’: ‘l1’ and ‘C’: 100000. So, the final obtained best parameters are: {'C': 100000, 'max_features': None, 'norm': 'l1', 'sublinear_tf': False}, with a best F1 score of 0.544.

Performance comparison with BERT

In this section, I will:

(a) Encode the text using the ‘feature-extraction’ pipeline from the HuggingFace library with the ‘roberta_base’ model. Pass the context vectors (without any other previous features) into a LogisticRegression classifier from scikit-learn and train using the training set.

(b) Train an end-to-end classifier using the ‘trainer’ function from the HuggingFace library, again using the ‘roberta_base’ model. Using a learning rate = 1e-4, epochs = 1, batch_size = 16 and no weight decay.

(c) Try different values for the model, learning_rate, epochs and batch_size.

Conclusion

The model has sufficiently high accuracy performance for the stated purpose of the classifier. I am able to get satisfactory values for all metrics.
The deployment of this model cannot have any negative societal effects, in fact it can be used not only for reviews, but other text related classifications as well.
Further steps that could be taking to improve the classification:

Data samples could be bigger: More data could improve performance by providing more training set.
Data could be tokenized more efficiently, I tokenized the data but I could still not improve the html tags present in my text files. So, that could be improved.



