# Code Submission Overview

<h3> I. Experiments </h3>

The code submission contains three experiments. Each is run separately by running the following programs:

- experiment_main.py
This is the main experiment. Used to generate the submissions on kaggle.

- experiment_2.py
Tests to see if majority ratio and decision date effect the performance of predictions.

- experiment_3.py
Tests to see the effects of different imputation techniques on issue area.

<h3> II. Folders </h3>

Below are important folders where results of experiments are stored.

<h5> Prediction </h5>
- Stores the csv prediction for the test set. This were uploaded to Kaggle for predictive evaluation.

<h5> Saved Models </h5>
- The model configuration after training are saved in a pickle (pkl) file. Subsequent runs of the experiments will simply load the pickle files and
bypass retraining the model again, saving time. If you want to retrain the model from scratch delete the pickle file and rerun the experiment programs.

<h5> Additional Details </h5>
- KBestScores: stores a ranked list of best features evaluated by the KBestFeatures function for each experiment. 
- RandomForestBestFeatures: stores the a ranked list of best features evaluated by Random Forest classifier.


<h3> Steps to the Run Main Experiment </h3>

1) Run experiment_main.py

2) The models will start training. This might take up to 10 minutes depending on your device specifications.

3) The test set predictions for each model will be saved in the Predictions folder. This will be named as [model_name]_Main.csv
The folder starts with the file generated already. If you deleted the file, you can run the program to generate them again.

4) The model details will be stored in the pickle file as [model_name]_Main.pkl.

The folders starts with the file generated. If you deleted the file, you can run the program to generate them again.

5) Additional details will save other information like Best Feature scores (KBestScores_Main.csv) 
and Best Random Forest features (RandomForestBestFeatures_Main.csv).


<h3> Run Experiments 2 or 3 </h3>

1) Run experiment_2.py or experiment_3.py

2) The models will start training. This might take up to 10 minutes depending on your device specifications.

3) The test set predictions for each model will be saved in the Predictions folder. This will be named as [model_name]_[V2/V3].csv.
The folder starts with the file generated already. If you deleted the file, you can run the program to generate them again.

4) The model details will be stored in the pickle file as [model_name]_[V2/V3].pkl.
The folders starts with the file generated. If you deleted the file, you can run the program to generate them again.

5) Additional details will save other information like Best Feature scores (KBestScores_[V2/V3].csv) 
and Best Random Forest features (RandomForestBestFeatures_[V2/V3].csv).
