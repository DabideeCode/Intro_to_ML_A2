import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

def write_prediction_to_file(df, file_name):
    df.to_csv(file_name, index=False, quoting=csv.QUOTE_ALL)

def sembed_union(df, file_path):
    numpy = np.load(file_path)

    conv_row = np.size(numpy, axis=0)
    conv_col = np.size(numpy, axis=1)

    x = range(conv_row)
    y = range(conv_col)
    sembed_column_names = []

    for j in y:
            sembed_column_names.append("conv{}".format(j+1))
    # Returns dataframe of sentence embedding
    conv_df = pd.DataFrame(numpy, columns=sembed_column_names)
    print(type(conv_df))
    print(type(df))
    return pd.concat([df, conv_df], sort=False, axis=1)

def save_model(experiment_name,file_name, model):
    path = "Saved Models/"
    with open(path+file_name+"_"+experiment_name+".pkl", 'wb') as file:
        pickle.dump(model, file)

def load_model(experiment_name, file_name):
    path = "Saved Models/"
    file_path = Path(path+file_name+"_"+experiment_name+".pkl")
    
    if file_path.exists():
        with open(path+file_name+"_"+experiment_name+".pkl", 'rb') as file:
            return pickle.load(file)
    else:
        return None
    
def evaluate_models(models, X, y):
    for model_name, model in models.items():
        print(f"Results for {model_name}:")
        y_pred = model.predict(X)
        # cm = confusion_matrix(y, y_pred, labels=model.classes_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        # disp.plot(cmap='Blues')
        # plt.title(model_name)
        # plt.show()
        print(classification_report(y, y_pred, zero_division=0))
        print(f"Best parameters: {model.get_params()}")
        print("="*60)


def create_best_classifiers(experiment_name, models, param_grid, X, y):
    best_models = {}
    for model_name, model in models.items():
        if not load_model(experiment_name, model_name):
            print("Training: ", model_name)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name], 
                                    cv=StratifiedKFold(n_splits=5), scoring='accuracy', n_jobs=-1, refit=True)
            grid_search.fit(X, y)
            best_models[model_name] = grid_search.best_estimator_
            save_model(experiment_name, model_name, grid_search.best_estimator_)
            results = grid_search.cv_results_
            
            for mean_score, std_score, params in zip(results["mean_test_score"], results["std_test_score"], results["params"]):
                print(f"Mean Score: {mean_score:.3f}, Std Score: {std_score:.3f}, Parameters: {params}")
        else:
            print("Exists: ", model_name)
            best_models[model_name] = load_model(experiment_name, model_name)
    
    # Evaluate models
    print("============== Results for Predicting Training Labels ==============")
    evaluate_models(best_models, X, y)

    return best_models

def predict_test(models, case_id, X, suffix, record):
    file_path = 'Predictions/'
    file_type = '.csv'
    for model_name, model in models.items():
        pred = model.predict(X)
        output_df = pd.DataFrame({
            'case_id': case_id,  # Replace 'Index' with a unique identifier if available
            'successful_appeal': pred
        })
        if record:
            output_df.to_csv(file_path+model_name+"_"+suffix+file_type, index=False)
    return None

def extract_datefields(df):
    date_fields = ['argument_date']
    
    df[date_fields] = df[date_fields].apply(pd.to_datetime)
    
    df['argument_year'] = df['argument_date'].dt.year
    df['argument_month'] = df['argument_date'].dt.month
    df['argument_day'] = df['argument_date'].dt.day
    
    return df

def extract_datefields_difference(df):
    date_fields = ['argument_date', 'decision_date']
    
    df[date_fields] = df[date_fields].apply(pd.to_datetime)
    
    df['argument_year'] = df['argument_date'].dt.year
    df['argument_month'] = df['argument_date'].dt.month
    df['argument_day'] = df['argument_date'].dt.day

    df['decision_year'] = df['decision_date'].dt.year
    df['decision_month'] = df['decision_date'].dt.month
    df['decision_day'] = df['decision_date'].dt.day

    df['days_until_decision'] = (df['decision_date'] - df['argument_date']).dt.days
    
    return df