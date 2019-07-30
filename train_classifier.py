import sys
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sqlite3
import nltk
import pickle
import re
import numpy as np
nltk.download(['punkt', 'wordnet','stopwords'])

# import statements
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from scipy import sparse as sp_sparse
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics import recall_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,fbeta_score

def load_data(database_filepath):
    engine = create_engine(database_filepath)
    df = pd.read_sql("SELECT * FROM cleaned_df", engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    return X,Y

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    clean_tokens = []
    clean_token = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        clean_token = [w for w in clean_tokens if not w in stop_words]

    return clean_token
stop_words = set(stopwords.words("english"))
def dummy_fun(doc):
    return doc
def tfidf_features(X_train, X_test,dummy_fun):
        """
            X_train, X_val, X_test — samples        
            return TF-IDF vectorized representation of each sample and vocabulary
        """
        # Create TF-IDF vectorizer with a proper parameters choice
        # Fit the vectorizer on the train set
        # Transform the train, test, and val sets and return the result
        tfidf_vectorizer = TfidfVectorizer(analyzer='word',stop_words=stop_words,ngram_range=(1,2),
                      max_df=0.9,min_df=5,tokenizer=dummy_fun,preprocessor=dummy_fun,token_pattern=None)   
        X_train=tfidf_vectorizer.fit_transform(X_train)
        X_test=tfidf_vectorizer.transform(X_test)
        return X_train, X_test, tfidf_vectorizer.vocabulary_
 
def build_model(X_train, y_train):
        estimator= LogisticRegression(random_state=0, solver='lbfgs',
                                      multi_class='multinomial')
        multi_class_lm=OneVsRestClassifier(estimator)
        multi_target_lm = MultiOutputClassifier(multi_class_lm)
        model=multi_target_lm.fit(X_train,y_train)
        return model


def evaluate_model(model, X_test_tfidf, y_test):
    lm_predictions=pd.DataFrame(model.predict(X_test_tfidf),columns=[y_test.columns])
    for column in y_test.columns:
        for y_true, predicted in zip([lm_predictions.loc[:,column]],[y_test.loc[:,column]]):
            print(classification_report(y_true, predicted))


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X=X.apply(lambda x:tokenize(x))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_train_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_test, dummy_fun)
        print('Building model...')
        model = build_model(X_train_tfidf,y_train)
        
#         print('Training model...')
#         model.fit(X_train_tfidf, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test_tfidf, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()