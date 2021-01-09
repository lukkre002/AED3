import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
#Zadanie 1
X_train = numpy.loadtxt('X_train.txt', delimiter=' ')
X_test = numpy.loadtxt('X_test.txt', delimiter=' ')
Y_train = numpy.loadtxt('y_train.txt')
Y_test = numpy.loadtxt('y_test.txt')
#Zadanie 2
classifiers = {'svm': svm.SVC(), 'knn': KNeighborsClassifier(), 'dt': DecisionTreeClassifier(), 'rf': RandomForestClassifier()}

predictions = {}
for key in classifiers:
    classifiers[key].fit(X_train, Y_train)
    predictions[key] = classifiers[key].predict(X_test)

#Zadanie 3
def zad3(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


for key in predictions:
    print('\nWynik dla:  ' + key)
    print('Precyzja: {}'.format(accuracy_score(Y_test, predictions[key])))
    print('RS micro: {}'.format(recall_score(Y_test, predictions[key], average='micro')))
    print('RS macro: {}'.format(recall_score(Y_test, predictions[key], average='macro')))
    print('F1 micro: {}'.format(f1_score(Y_test, predictions[key], average='micro')))
    print('F1 macro: {}'.format(f1_score(Y_test, predictions[key], average='macro')))
    print("AUC micro: {}".format(zad3(Y_test, predictions[key], average='micro')))
    print("AUC macro: {}".format(zad3(Y_test, predictions[key], average='macro')))
    plt.figure(num='Confusion matrix: ' + key)
    seaborn.heatmap(confusion_matrix(Y_test, predictions[key]), annot=True, fmt='d')
    plt.show()

#Zadanie 4
class Score(object):
    def __init__(self, value):
        self.value = value
        self.avg = numpy.mean(value)
        self.std = numpy.std(value)


scores = {}
for key in classifiers:
    score = Score(cross_val_score(classifiers[key], X_train, Y_train, cv=5))
    scores[key] = score
    print('\nMetoda  ' + key)
    print('Wynik krzyżowy: {}'.format(score.value))
    print('Średnia : {}'.format(score.avg))
    print('Odchylenie standardowe: {}'.format(score.std))

print('\nNajlepsze wyniki: ')
print('maksymalna średnia {}'.format(max(scores, key=lambda k: scores[k].avg)))
print('minimalne średnie odchylenie  {}'.format(min(scores, key=lambda k: scores[k].std)))
