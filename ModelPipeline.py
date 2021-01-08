from DataPreProcessing import DataPreProcessing
from MachineLearningModels import MachineLearningModels
from DeepNeuralNetModels import DeepNeuralNetModels
from ModelEvaluation import ModelEvaluation


#Data Pre-Processing
data_processing = DataPreProcessing()
ml_model=MachineLearningModels()
dl_model=DeepNeuralNetModels()
model_eval = ModelEvaluation()
file_name='datascience.jsonl'
data_corpus=data_processing.prepareDataCorpus(file_name)
X_train,X_test,y_test,y_train = data_processing.prepareTrainTestData(file_name,data_corpus)


#Machine Learning and Stat Models
model_name="Logistic"
lr_classifier,y_pred_lr = ml_model.trainLogisticRegression(X_train,X_test,y_test,y_train)
model_eval.calculateMLModelScores(model_name,y_test,y_pred_lr)

model_name="Random Forest"
rf_classifier,y_pred_rf = ml_model.trainRandomForest(X_train,X_test,y_test,y_train)
model_eval.calculateMLModelScores(model_name,y_test,y_pred_rf)

model_name="XG Boost"
xg_classifier, y_pred_xg = ml_model.trainXGBoost(X_train,X_test,y_test,y_train)
model_eval.calculateMLModelScores(model_name,y_test,y_pred_xg)

model_name="Naive Bayes"
nb_classifier, y_pred_nb = ml_model.trainNaiveBayes(X_train,X_test,y_test,y_train)
model_eval.calculateMLModelScores(model_name,y_test,y_pred_nb)

model_name="SVM"
svm_classifier, y_pred_svm = ml_model.trainSVMClassifier(X_train,X_test,y_test,y_train)
model_eval.calculateMLModelScores(model_name,y_test,y_pred_svm)


# Deep Learning based Neural Net Models
X_train,X_test,y_test,y_train = data_processing.prepareDataForDeepModels(file_name)
dl_model.trainDenseNets(X_train,X_test,y_test,y_train)
dl_model.trainDeepModelOnEmbeddings(X_train,X_test,y_test,y_train)
dl_model.trainCNNModelOnEmbeddings(X_train,X_test,y_test,y_train)