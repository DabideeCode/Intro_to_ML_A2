import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

# Loading corpus

nltk_data_path = os.path.expanduser('~') + '/nltk_data/'

if not os.path.exists(nltk_data_path + 'corpora/stopwords'):
    nltk.download('stopwords', quiet=True)  # 'quiet=True' to suppress messages
if not os.path.exists(nltk_data_path + 'corpora/wordnet'):
    nltk.download('wordnet', quiet=True)

stop_words = (stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Prevent warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Loading datasets

df = pd.read_json("data/train.jsonl", lines=True)
df_dev = pd.read_json("data/dev.jsonl", lines=True)
df_test = pd.read_json("data/test.jsonl", lines=True)


# Data setup

use_features_list = ["petitioner_category", "respondent_category", 
                    "issue_area", "year", "court_hearing_length", "utterances_number", "argument_date", "decision_date", "majority_ratio"]
train_df = df[use_features_list]
train_labels = df['successful_appeal']

eval_df = df_dev[use_features_list]
eval_labels = df_dev['successful_appeal']

test_df = df_test[use_features_list]
test_df_id = df_test['case_id']


# Impute issue area with most frequent value

imp_majority = SimpleImputer(missing_values="UNKNOWN", strategy="most_frequent")

train_df.loc[:, 'issue_area'] = imp_majority.fit_transform(train_df[['issue_area']])
eval_df.loc[:, 'issue_area'] = imp_majority.transform(eval_df[['issue_area']])
test_df.loc[:, 'issue_area'] = imp_majority.transform(test_df[['issue_area']])

# Encode issue area

ohe = OneHotEncoder(sparse_output=False).set_output(transform='pandas')

ohetransform = ohe.fit_transform(train_df[['issue_area']])
train_df = pd.concat([train_df, ohetransform], axis = 1).drop(columns = ['issue_area'] )

ohetransform = ohe.transform(eval_df[['issue_area']])
eval_df = pd.concat([eval_df, ohetransform], axis = 1).drop(columns = ['issue_area'] )

ohetransform = ohe.transform(test_df[['issue_area']])
test_df = pd.concat([test_df, ohetransform], axis = 1).drop(columns = ['issue_area'] )

# Text Preprocessing

petcat_vector_training = train_df['petitioner_category']
rescat_vector_training = train_df['respondent_category']

petcat_vector_eval = eval_df['petitioner_category']
rescat_vector_eval = eval_df['respondent_category']

petcat_vector_test = test_df['petitioner_category']
rescat_vector_test = test_df['respondent_category']

analyzer = TfidfVectorizer().build_analyzer()

def lemmatized_words(doc):
    return (lemmatizer.lemmatize(w) for w in analyzer(doc) if w not in stop_words)

tfid_vectorizer = TfidfVectorizer(lowercase=True, analyzer=lemmatized_words, max_features=100)

tfid_petcat_training = tfid_vectorizer.fit_transform(petcat_vector_training)
petcat_col_training = pd.DataFrame(tfid_petcat_training.toarray(), columns=tfid_vectorizer.get_feature_names_out()).add_prefix("petcat_")
tfid_petcat_eval = tfid_vectorizer.transform(petcat_vector_eval)
petcat_col_eval = pd.DataFrame(tfid_petcat_eval.toarray(), columns=tfid_vectorizer.get_feature_names_out()).add_prefix("petcat_")
tfid_petcat_test = tfid_vectorizer.transform(petcat_vector_test)
petcat_col_test = pd.DataFrame(tfid_petcat_test.toarray(), columns=tfid_vectorizer.get_feature_names_out()).add_prefix("petcat_")

tfid_rescat_training = tfid_vectorizer.fit_transform(rescat_vector_training)
rescat_col_training = pd.DataFrame(tfid_rescat_training.toarray(), columns=tfid_vectorizer.get_feature_names_out()).add_prefix("rescat_")
tfid_rescat_eval = tfid_vectorizer.transform(rescat_vector_eval)
rescat_col_eval = pd.DataFrame(tfid_rescat_eval.toarray(), columns=tfid_vectorizer.get_feature_names_out()).add_prefix("rescat_")
tfid_rescat_test = tfid_vectorizer.transform(rescat_vector_test)
rescat_col_test = pd.DataFrame(tfid_rescat_test.toarray(), columns=tfid_vectorizer.get_feature_names_out()).add_prefix("rescat_")

train_df = pd.concat([train_df, petcat_col_training, rescat_col_training], axis=1).drop(columns = ['respondent_category','petitioner_category'] )
eval_df = pd.concat([eval_df, petcat_col_eval, rescat_col_eval], axis=1).drop(columns = ['respondent_category','petitioner_category'] )
test_df = pd.concat([test_df, petcat_col_test, rescat_col_test], axis=1).drop(columns = ['respondent_category','petitioner_category'] )

# Feature engineer using argument date field

train_df = utils.extract_datefields_difference(train_df).drop(columns = ['argument_date', 'decision_date'])
eval_df = utils.extract_datefields_difference(eval_df).drop(columns = ['argument_date', 'decision_date'])
test_df = utils.extract_datefields_difference(test_df).drop(columns = ['argument_date', 'decision_date'])

# Impute using iterative imputer

iter_imputer = IterativeImputer(max_iter=20, random_state=1)

train_array = iter_imputer.fit_transform(train_df)
eval_array = iter_imputer.transform(eval_df)
test_array = iter_imputer.transform(test_df)

train_df = pd.DataFrame(train_array, columns=train_df.columns)
eval_df = pd.DataFrame(eval_array, columns=eval_df.columns)
test_df = pd.DataFrame(test_array, columns=test_df.columns)

# Feature selection

selector = SelectKBest(f_classif, k=200)
kbest = selector.fit(train_df, y=train_labels)
selected_features = train_df.columns[kbest.get_support()]

train_df = pd.DataFrame(kbest.transform(train_df), columns=selected_features)
eval_df = pd.DataFrame(kbest.transform(eval_df), columns=selected_features)
test_df = pd.DataFrame(kbest.transform(test_df), columns=selected_features)


# Display feature scores

feature_scores = selector.scores_
scores = pd.DataFrame(columns=['Feature', 'Score'])
for feature, score in zip(train_df.columns, feature_scores):
    row = pd.DataFrame([{'Feature' : feature, 'Score' : score }])
    scores = pd.concat([scores, row], ignore_index=True)

scores = scores.sort_values(by='Score', ascending=False)
scores.to_csv("Additional Details/KBestScores_V2.csv", index=False)


# Normalize features

feature_scaler = StandardScaler()

X_scaled_train = feature_scaler.fit_transform(train_df)
y_scaled_train = train_labels

X_scaled_eval = feature_scaler.transform(eval_df)
y_scaled_eval = eval_labels

X_scaled_test = feature_scaler.transform(test_df)


# Zero-R Classifier

zero_R_clf = DummyClassifier(strategy="most_frequent")
zero_R_clf.fit(X_scaled_train, y_scaled_train)

zero_R_pred_y = zero_R_clf.predict(X_scaled_eval)
zero_R_accuracy = accuracy_score(y_pred = zero_R_pred_y, y_true = y_scaled_eval)

print("Zero-R Accuracy Training: ",(zero_R_accuracy * 100))


# Models to be trained

models = {
    'RandomForest': RandomForestClassifier(random_state=1),
    'MLPClassifier': MLPClassifier(random_state=1, learning_rate_init=0.01, early_stopping=True)
}

# Parameter grid for hyperparameter optimization

param_grid = {
    'RandomForest': {'n_estimators': [50, 100, 200, 300], 'max_depth': [5, 10, 20, 30]},
    'MLPClassifier': {'max_iter': [500, 700, 900], 'activation': ['relu','logistic','tanh'], 'alpha': [0.001, 0.01, 0.1], 
                      'hidden_layer_sizes': [(50, 50), (50, 100), (100, 100, 100)], 'solver': ['adam', 'sgd'], 'learning_rate': ['constant', 'adaptive']}
}


# Train models
best_classifiers = utils.create_best_classifiers("V2", models, param_grid, X_scaled_train, y_scaled_train)

# Save feature importance for Random Forest

feature_importances = best_classifiers['RandomForest'].feature_importances_

# Create a DataFrame to view feature importance scores

importance_df = pd.DataFrame({
    'Feature': train_df.columns,
    'Importance': feature_importances
})

importance_df = importance_df.sort_values(by='Importance', ascending=False)
importance_df.to_csv("Additional Details/RandomForestBestFeatures_V2.csv", index=False)


# Test models with the evaluation set
print("============== Results for Predicting Evaluation Labels ==============")
utils.evaluate_models(best_classifiers, X_scaled_eval, y_scaled_eval)


# Predict the class for the test set and record them in a csv file

utils.predict_test(best_classifiers, test_df_id, X_scaled_test, "V2", record=True)