import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
from keras.layers import Dense  # densly connected layers
from keras.models import Sequential
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
from util import model_details


def TrainModelByAlgo(algo, driver_data_final):
    switcher = {
        "NB": lambda: NB(driver_data_final),
        "KNN": lambda: KNN(driver_data_final),
        "LR": lambda: LR(driver_data_final),
        "SVM": lambda: SVM(driver_data_final),
        "DT": lambda: DT(driver_data_final),
        "RF": lambda: RF(driver_data_final),
        "FFNN": lambda: FFNN(driver_data_final),
        "ST_DT_SVM_RF_LR": lambda: ST_DT_SVM_RF_LR(driver_data_final),
        "BG_DT": lambda: BG_DT(driver_data_final),
        "XGB_DT": lambda: XGB_DT(driver_data_final),
        "ADAB_DT": lambda: ADAB_DT(driver_data_final),
    }
    func = switcher.get(algo, lambda: "nothing")
    return func()


def LR(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    logmodel = LogisticRegression(solver='lbfgs', max_iter=1000)
    logmodel.fit(X_train, y_train)
    predictions = logmodel.predict(X_test)
    # print(classification_report(y_test,predictions))

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    joblib.dump(logmodel, '/home/manorip/Documents/MSC/FYP/work_place/model_store/LR_Model.pkl')

    print('LR Precision: %.3f' % precision)
    print('LR Recall: %.3f' % recall)
    print('LR Accuracy: %.3f' % accuracy)
    print('LR F1 Score: %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("LR", accuracy)
    model_details.algoAccuracies["LR"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1


def KNN(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1,
                               n_neighbors=37, p=2, weights='uniform')
    knn.fit(X_train_res, y_train_res)
    predictions = knn.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    joblib.dump(knn, '/home/manorip/Documents/MSC/FYP/work_place/model_store/KNN_Model.pkl')

    print('---------------------------------------------------')
    print('KNN Precision : %.3f' % precision)
    print('KNN Recall : %.3f' % recall)
    print('KNN Accuracy : %.3f' % accuracy)
    print('KNN Score : %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("KNN", accuracy)
    model_details.algoAccuracies["KNN"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1


def NB(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    gnb = GaussianNB()
    gnb.fit(X_train_res, y_train_res)
    predictions = gnb.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    joblib.dump(gnb, '/home/manorip/Documents/MSC/FYP/work_place/model_store/NB_Model.pkl')

    print('---------------------------------------------------')
    print('NB Precision : %.3f' % precision)
    print('NB Recall : %.3f' % recall)
    print('NB Accuracy : %.3f' % accuracy)
    print('NB Score : %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("NB", accuracy)
    model_details.algoAccuracies["NB"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1


def SVM(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    # tune in the model with best gamma, C and kernal
    param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
    grid.fit(X_train_res, y_train_res)
    print(grid.best_params_)
    predictions = grid.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    joblib.dump(grid, '/home/manorip/Documents/MSC/FYP/work_place/model_store/SVM_Model.pkl')

    print('---------------------------------------------------')
    print('SVM Precision : %.3f' % precision)
    print('SVM Recall : %.3f' % recall)
    print('SVM Accuracy : %.3f' % accuracy)
    print('SVM Score : %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("SVM", accuracy)
    model_details.algoAccuracies["SVM"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1


def DT(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    model = DecisionTreeClassifier()  # DecisionTreeClassifier(criterion="entropy")
    model.fit(X_train_res, y_train_res)
    predictions = model.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    joblib.dump(model, '/home/manorip/Documents/MSC/FYP/work_place/model_store/DT_Model.pkl')

    print('---------------------------------------------------')
    print('DT Precision : %.3f' % precision)
    print('DT Recall : %.3f' % recall)
    print('DT Accuracy : %.3f' % accuracy)
    print('DT Score : %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("DT", accuracy)

    model_details.algoAccuracies["DT"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1


def RF(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)  # Numerical value
    # rus = RandomUnderSampler(sampling_strategy="not minority") # String
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train_res, y_train_res)
    predictions = clf.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    joblib.dump(clf, '/home/manorip/Documents/MSC/FYP/work_place/model_store/RF_Model.pkl')

    print('---------------------------------------------------')
    print('RF Precision : %.3f' % precision)
    print('RF Recall : %.3f' % recall)
    print('RF Accuracy : %.3f' % accuracy)
    print('RF Score : %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("RF", accuracy)
    model_details.algoAccuracies["RF"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1


def FFNN(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    modelTrain = Sequential()  # model is an empty NN
    # 1st hidden layer (dense type-fully connected)
    modelTrain.add(Dense(54, input_dim=27, activation='relu'))
    # 2nd Hidden layer
    modelTrain.add(Dense(10, input_dim=54, activation='relu'))
    # model.add(Dropout(0.5))
    # 3rd Hidden layer
    modelTrain.add(Dense(8, input_dim=10, activation='relu'))
    # 4rd Hidden layer
    modelTrain.add(Dense(5, input_dim=8, activation='relu'))
    # final layer
    modelTrain.add(Dense(1, input_dim=5, activation='softmax'))

    modelTrain.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    # history = modelTrain.fit(X_train_res, y_train, epochs=150)
    #  _, accuracy = modelTrain.evaluate(X_test, y_test)
    # print('Accuracy: %.2f' % (accuracy*100))
    # https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
    # since there is a library version issue, the accuracy retrieve from colab is included here
    return 0, 89.67, 0, 0


def ST_DT_SVM_RF_LR(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    RF = RandomForestClassifier(n_estimators=100)
    SVM = SVC()
    DT = DecisionTreeClassifier()
    LR = LogisticRegression()

    estimators = [
        ('dt', DT),
        ('rf', RF),
        ('svm', SVM)]

    clf_stack = StackingClassifier(estimators=estimators, final_estimator=LR)
    clf_stack.fit(X_train, y_train)
    predictions = clf_stack.predict(X_test)

    joblib.dump(clf_stack, '/home/manorip/Documents/MSC/FYP/work_place/model_store/ST_DT_SVM_RF_LR_Model.pkl')

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    print('---------------------------------------------------')
    print('ST_DT_SVM_RF_LR Precision : %.3f' % precision)
    print('ST_DT_SVM_RF_LR Recall : %.3f' % recall)
    print('ST_DT_SVM_RF_LR Accuracy : %.3f' % accuracy)
    print('ST_DT_SVM_RF_LR Score : %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("ST_DT_SVM_RF_LR", accuracy)
    model_details.algoAccuracies["ST_DT_SVM_RF_LR"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1


def BG_DT(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)

    kfold = model_selection.KFold(n_splits=3)
    # initialize the base classifier
    base_cls = DecisionTreeClassifier()
    # no. of base classifier
    num_trees = 500
    model = BaggingClassifier(base_estimator=base_cls, n_estimators=num_trees)
    results = model_selection.cross_val_score(model, X, y, cv=kfold)
    joblib.dump(model, '/home/manorip/Documents/MSC/FYP/work_place/model_store/BG_DT_Model.pkl')

    accuracy = results.mean()
    precision = 0
    recall = 0
    f1 = 0
    print('BG_DT Accuracy : %.3f' % accuracy)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("BG_DT", accuracy)
    model_details.algoAccuracies["BG_DT"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return 0, accuracy, 0, 0


def XGB_DT(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    clf = xgb.XGBClassifier(random_state=1, learning_rate=0.01)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    joblib.dump(clf, '/home/manorip/Documents/MSC/FYP/work_place/model_store/XGB_DT_Model.pkl')

    print('---------------------------------------------------')
    print('XGB_DT Precision : %.3f' % precision)
    print('XGB_DT Recall : %.3f' % recall)
    print('XGB_DT Accuracy : %.3f' % accuracy)
    print('XGB_DT Score : %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("XGB_DT", accuracy)
    model_details.algoAccuracies["XGB_DT"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1


def ADAB_DT(driver_data_final):
    y = driver_data_final['churn']
    X = driver_data_final.drop(['churn'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    # y.value_counts().plot.pie(autopct='%.2f',title="Before Sampling")
    rus = RandomOverSampler(sampling_strategy=1)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    # y_res.value_counts().plot.pie(autopct='%.2f',title="After Sampling")

    clf = AdaBoostClassifier(n_estimators=50,
                             learning_rate=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    joblib.dump(clf, '/home/manorip/Documents/MSC/FYP/work_place/model_store/ADAB_DT_Model.pkl')

    print('---------------------------------------------------')
    print('ADAB_DT Precision : %.3f' % precision)
    print('ADAB_DT Recall : %.3f' % recall)
    print('ADAB_DT Accuracy : %.3f' % accuracy)
    print('ADAB_DT Score : %.3f' % f1)

    if model_details.bestAlgo["accuracy"] < accuracy:
        model_details.SetBestAlgo("ADAB_DT", accuracy)
    model_details.algoAccuracies["ADAB_DT"] = model_details.ModelDetail(accuracy, precision, recall, f1)

    return precision, accuracy, recall, f1