# TweetAnalysis
Work with a set of Tweets about US airlines and examine their sentiment polarity.The aim is to learn to classify Tweets as either “positive”, “neutral”, or “negative” by using two classifiers and pipelines for pre-processing and model building.

Below are the steps of the project:
1. Loading: First step is to define an input argument that defines the path from which to load
the dataset. After that, you will need to remove rows where the text field is null.
2. Pre-Processing: You will start by creating a pre-processing pipeline with the following stages:
• Tokenizer: Transform the text column by breaking down the sentence into words • Stop Word Remover: Remove stop-words from the words column
Hint: Use the import org.apache.spark.ml.feature.StopWordsRemover class.
• Term Hashing: Convert words to term-frequency vectors
Hint: Use the import org.apache.spark.ml.feature.HashingTF class
• Label Conversion: The label is a string e.g. “Positive”, which you need to convert to numeric format
Hint: Use the import org.apache.spark.ml.feature.StringIndexer class
Remember that you need to create a pipeline of the above steps and then transform the raw input dataset to a pre-processed dataset.
3. Model Creation - You will need to create two classification models that you can select from the MLlib classification library. You will have to create a ParameterGridBuilder for parameter tuning and then use the CrossValidator object for finding the best model. An example of this can be seen here: https://spark.apache.org/docs/2.2.0/api/scala/index.html#org. apache.spark.ml.tuning.CrossValidator
4. Model Testing & Evaluation: Next, you will create a random sample of the dataset and apply your model on it and output classification evaluation metrics, such as accuracy, etc. You can see details of multi-class evaluation metrics at https://spark.apache.org/docs/2.2.0/ mllib-evaluation-metrics.html.
5. Output: Finally, you have to write the output the classification metrics to a file whose location is specified by the second argument to the class.