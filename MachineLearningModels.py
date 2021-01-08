import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class MachineLearningModels:
    def __init__(self):
        pass

    def trainLogisticRegression(self,X_train,y_train,X_test,y_test):
        classifier = LogisticRegression()
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print("Logistic Predicition=" + str(classifier.predict(X_test[2])))
        score = classifier.score(X_test, y_test)
        print("Logistic score= " + str(score))
        return classifier,y_pred

    def trainRandomForest(self, X_train, y_train, X_test,y_test):
        rf_classifier = RandomForestClassifier()
        rf_classifier.fit(X_train, y_train)
        y_pred = rf_classifier.predict(X_test)
        print("Random Forest Predicition=" + str(rf_classifier.predict(X_test[2])))
        rf_score = rf_classifier.score(X_test, y_test)
        print("Random Forest score= " + str(rf_score))
        return rf_classifier, y_pred

    def trainXGBoost(self, X_train, y_train, X_test,y_test):
        xg_classifier = xgb.XGBClassifier()
        xg_classifier.fit(X_train, y_train)
        y_pred = xg_classifier.predict(X_test)
        print("XGB predicition=" + str(xg_classifier.predict(X_test[2])))
        xg_score = xg_classifier.score(X_test, y_test)
        print("XGB score= " + str(xg_score))
        return xg_classifier, y_pred

    def trainNaiveBayes(self, X_train, y_train, X_test,y_test):
        Naive = naive_bayes.MultinomialNB()
        Naive.fit(X_train, y_train)
        y_pred = Naive.predict(X_test)
        print("Naive Bayes Accuracy Score -> ", accuracy_score(y_pred, y_test) * 100)
        return Naive, y_pred

    def trainSVMClassifier(self,X_train, y_train, X_test,y_test):
        SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
        SVM.fit(X_train, y_train)
        y_pred = SVM.predict(X_test)
        predictions_SVM = SVM.predict(X_test)
        print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, y_test) * 100)
        return SVM, y_pred

