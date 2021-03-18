import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
# Keras newural network dependencies
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Activation
import keras.backend as K
from keras.initializers import Constant
from keras.optimizers import Adam
# Other model dependencies
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# Preprocessing dependencies
from collections import Counter
import re
import string
import nltk
from nltk.corpus import stopwords

# Requirements.txt to install required if necessary

class CommentClassifier():
    def __init__(self, category, classifier, classifierType):
        self.category = category
        self.classifier = classifier
        self.classifierType = classifierType

    def setScore(self, score):
        self.score = score

# ---------------------------------------------

# For text cleaning
def remove_punct(input_text):
    result = input_text.translate(str.maketrans("", "", string.punctuation))
    return result

def remove_URL(input_text):
    result = re.sub(r"https?://\S+|www\.\S+","", input_text)
    return result

def remove_html(input_text):
    result = re.sub(r"<.*?>","",input_text)
    return result

def remove_tokens(input_text):
    result = re.sub(r"\n|\r|\t", "", input_text)
    return result

def remove_stopwords_nltk(text, stop):

    text = [word.lower() for word in text.split() if word.lower() not in stop]
    return " ".join(text)

# ---------------------------------------------

def word_counter(text):
    unique_word_counter = Counter()
    word_counts = []
    for i in text.values:
        text_word_count = 0
        for word in i.split():
            text_word_count += 1
            unique_word_counter[word] += 1
        word_counts.append(text_word_count)
    return word_counts, unique_word_counter

def init_params(labels):
    class_biases = []
    class_weights = []

    for (columnName, columnData) in labels.iteritems():
        neg, pos = np.bincount(columnData)
        total = neg + pos
        class_prop = pos/total
        print('Class {}:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
            columnName, total, pos, 100 * class_prop))
        bias = np.log([pos/neg])
        weight = 1/class_prop
        print(weight)
        print()
        class_biases.append(bias)
        class_weights.append(weight)

    return class_biases, class_weights
    
def fit_predict(features, target, targetName, classifier, classifierList, classifierType):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state=2)
    classifierObj = CommentClassifier(targetName, classifier, classifierType)
    print("Training ",targetName, " category ...")
    classifierObj.classifier.fit(X_train, y_train)
    validation_pred = classifierObj.classifier.predict_proba(X_test)

    scorer = tf.keras.metrics.AUC()
    scorer.update_state(validation_pred[:,1], y_test)
    validation_auc = (scorer.result().numpy())*100

    validation_acc = classifierObj.classifier.score(X_test, y_test)
    print()
    classifierList.append(classifierObj)

    return validation_auc, validation_acc


def predict_score(classifierObjList, features, labels, training_auc, training_acc):
    test_predictions = np.zeros(labels.shape)
    test_pred_classes = np.zeros(labels.shape)
    testing_auc_total = 0

    for index, classifierObj in enumerate(classifierObjList):
        y_pred = (classifierObj.classifier.predict_proba(features))
        y_pred_classes = (classifierObj.classifier.predict(features))
        testing_scorer_auc = tf.keras.metrics.AUC()
        testing_scorer_auc.update_state(y_pred[:,1], labels.iloc[:,index])
        auc_score = (testing_scorer_auc.result().numpy())*100
        testing_auc_total+= auc_score
        test_pred_classes[:,index] = y_pred_classes


    print("Validation auc_score of ", 
        classifierObj.classifierType ," = ", "{0:.4f}".format(training_auc), "%")

    print("Validation accuracy of ", 
        classifierObj.classifierType ," = ", "{0:.4f}".format(training_acc), "%")
    
    testing_scorer_acc = tf.keras.metrics.Accuracy()
    testing_scorer_acc.update_state(test_pred_classes, labels)
    testing_acc = (testing_scorer_acc.result().numpy())*100
    testing_auc = testing_auc_total/6

    print("Testing auc_score of ", 
        classifierObj.classifierType ," = ", "{0:.4f}".format(testing_auc), "%")

    print("Testing accuracy of ", 
        classifierObj.classifierType ," = ", "{0:.4f}".format(testing_acc), "%")

    


def main():
    df = pd.read_csv("train.csv")
    labels = df.drop(['id','comment_text'], axis=1)

    nltk.download('stopwords')
    stop = set(stopwords.words("english"))

    print("Cleaning text in the training data...")
    df["comment_text"].fillna("unknown", inplace=True)
    df["comment_text"] = df["comment_text"].map(lambda x: remove_stopwords_nltk(x, stop))
    df["comment_text"] = df["comment_text"].map(lambda x: remove_punct(x))
    df["comment_text"] = df["comment_text"].map(lambda x: remove_URL(x))
    df["comment_text"] = df["comment_text"].map(lambda x: remove_html(x))
    df["comment_text"] = df["comment_text"].map(lambda x: remove_tokens(x))

    # Converting comments into a sparse matrix of words
    print("Vectorizing training features ...")
    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(df.comment_text)

    # Real test data
    test_features_df = pd.read_csv("test.csv")
    test_label_df = pd.read_csv("test_labels.csv")
    test_combined_df = (pd.merge(test_features_df, test_label_df, left_on='id', right_on='id', how='left')) # Combining labels and features
    test_combined_df = test_combined_df[(test_combined_df.toxic != -1)]  # Removing records with -1 tags

    print("Cleaning text in the testing data...") # Cleaning testing data
    test_combined_df["comment_text"].fillna("unknown", inplace=True)
    test_combined_df["comment_text"] = test_combined_df["comment_text"].map(lambda x: remove_stopwords_nltk(x, stop))
    test_combined_df["comment_text"] = test_combined_df["comment_text"].map(lambda x: remove_punct(x))
    test_combined_df["comment_text"] = test_combined_df["comment_text"].map(lambda x: remove_URL(x))
    test_combined_df["comment_text"] = test_combined_df["comment_text"].map(lambda x: remove_html(x))
    test_combined_df["comment_text"] = test_combined_df["comment_text"].map(lambda x: remove_tokens(x))

    print("Vectorizing testing features ...")
    test_features = vectorizer.transform(test_combined_df.comment_text)
    test_labels = test_combined_df.drop(['id','comment_text'], axis=1)

    text = df.comment_text
    comment_word_counts, vocab_counter = word_counter(text)
    num_words = len(vocab_counter)
    # Max number of words in a sequence - for keras
    max_length = 60

    print("Tokenizing training features ...")
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(df.comment_text)
    word_index = tokenizer.word_index
    train_sequences = tokenizer.texts_to_sequences(df.comment_text)
    train_padded = pad_sequences(
            train_sequences, maxlen=max_length, padding="post", truncating="post"
            )
    print("Tokenizing testing features ...")
    test_sequences = tokenizer.texts_to_sequences(test_combined_df.comment_text)
    test_padded = pad_sequences(
        test_sequences, padding="post", truncating="post", maxlen=max_length
            )


    run=True
    while run:
        print()
        print('---------------------------------------------------------------------------')
        print()
        print('Welcome to the Toxic Comment Classifier!\n')
        print('Choose an option below:\n')
        print('1. View descriptive statistics on the text data')
        print('2. Run naive-bayes classifier')
        print('3. Run logistic regression classifier')
        print('4. Run LSTM multi-label classifier model')
        print('5. Exit\n')
        option = input('Option number: ')
    
        # define the function blocks
        def sum_stats():
            class_biases = []
            label_names = []
            class_proportions = []

            for (columnName, columnData) in df.drop(['id','comment_text'], axis=1).iteritems():
                neg, pos = np.bincount(columnData)
                total = neg + pos
                print('Class {}:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
                    columnName, total, pos, 100 * pos / total))
                bias = np.log([pos/neg])
                class_biases.append(bias)
                label_names.append(columnName)
                class_proportions.append(100*pos/total)

            class_biases = np.array(class_biases)
            label_names = np.array(label_names)
            class_proportions = np.array(class_proportions)

            x_pos = [i for i, _ in enumerate(label_names)]

            plt.bar(x_pos, class_proportions, color='indigo')
            plt.ylabel("Positive observations % of total", fontsize=16)
            plt.title("Class compositions", fontsize=20)
            plt.xticks(x_pos, label_names, fontsize=12.5)

            plt.show()

            comment_lengths = df.comment_text.str.len()
            comment_lengths.mean(), comment_lengths.std(), comment_lengths.max()
            comment_lengths.hist(color='mediumpurple', label='Comment Length (chars) Histogram')

        def clf_nb():
            print ("Naive Bayes Selected.\n")
            MultiNBClassifiers = []
            i = 0
            auc_total=0
            acc_total=0

            # Fitting and testing naive-bayes classifiers
            for (columnName, columnData) in labels.iteritems():
                print("Building naive_bayes classifier for ",columnName, " category ...")
                prior_negative = df[columnName].value_counts()[0]/df.shape[0]
                prior_positive = 1- prior_negative
                class_prior= [prior_negative, prior_positive]
                validation_auc, validation_acc = fit_predict(
                    train_features, columnData, columnName, 
                    MultinomialNB(class_prior=class_prior, fit_prior=True), 
                    MultiNBClassifiers, 'Naive-bayes'
                    )
                auc_total+=validation_auc
                acc_total+=validation_acc

            train_acc = (acc_total/6)*100
            train_auc = (auc_total/6)
            predict_score(MultiNBClassifiers, test_features, test_labels, train_auc, train_acc)

        def clf_lr():
            print ("Logistic Regression Selected\n")
            # Liblinear logistic regression
            LogisticRegressionModels_SAG = []
            i = 0
            X_train, X_test, y_train, y_test = train_test_split(train_features, labels, test_size = 0.3, random_state=2)
            auc_total=0
            acc_total=0

            # Fitting and testing naive-bayes classifiers
            for (columnName, columnData) in labels.iteritems():
                print("Building logistic regression classifier for ",columnName, " category ...")
                prior_negative = df[columnName].value_counts()[0]/df.shape[0]
                prior_positive = 1- prior_negative
                class_prior= [prior_negative, prior_positive]
                validation_auc, validation_acc = fit_predict(
                    train_features, columnData, columnName, 
                    LogisticRegression(C=4, dual=True, solver='liblinear', max_iter=100, class_weight='balanced'), 
                    LogisticRegressionModels_SAG, 'Logistic Regression'
                    )
                auc_total+=validation_auc
                acc_total+=validation_acc

            train_acc = (acc_total/6)*100
            train_auc = (auc_total/6)
            predict_score(LogisticRegressionModels_SAG, test_features, test_labels, train_auc, train_acc)
        

        def rnn_multi():
            # Getting biases and weights based on the distribution of the classes
            # This is important due to the imbalance in the classes
            
            biases, weights = init_params(labels)
            biases = np.array(biases)

            def bias_init(bias_shape, dtype='float64'):
                return K.variable(biases[:,0])

            class_weights_dict = {0: weights[0],
                      1: weights[1],
                      2: weights[2],
                      3: weights[3],
                      4: weights[4],
                      5: weights[5]}

            n_outputs=6 # 6 possible categories in this multi-label classification
            model = Sequential()
            model.add(Embedding(num_words, 32, input_length=max_length))
            model.add(LSTM(64, dropout=0.1))
            model.add(Dense(n_outputs, activation="sigmoid", bias_initializer=bias_init))
            optimizer = Adam(learning_rate=2e-3)
            metrics=['accuracy',tf.keras.metrics.AUC(multi_label=True), tf.keras.metrics.Precision()]

            model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=metrics)
            print(model.summary())
            history = model.fit(
            train_padded, labels, epochs=4, validation_split=0.2, class_weight=class_weights_dict
            )
            print("-------------------------------------------------------------------------------")
            print()
            print(model.evaluate(test_padded, test_labels))

        def invalid():
            print ("Invalid input, try again.\n")
            
        def exit():
            run = False

        if option == '5':
            break
        elif option == '1':
            sum_stats()
        elif option == '2':
            clf_nb()
        elif option == '3':
            clf_lr()
        elif option == '4':
            rnn_multi()
        else:
            invalid()
            continue
    

if __name__ == "__main__":
    main()
