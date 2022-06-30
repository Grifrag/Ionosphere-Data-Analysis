import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve,auc,confusion_matrix,accuracy_score,classification_report,plot_confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC




data = pd.read_csv('C:\\Users\\Grifrag\\PycharmProjects\\untitled\\venv\\ionosphere.csv',delimiter=';')
n_rows,n_cols = data.shape
#Preprocessing
#To deutero column den exei diakumnash,opote den mas prosferei kati sthn analush
data.drop(columns='Column2',inplace=True);

data.rename(columns={"Column35":"Value"},inplace=True)
print(data.head())
sns.countplot(x="Value",data=data)
plt.show()
X=data.iloc[:,:-1].to_numpy()
y=np.array(data.iloc[:,-1])

#train split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
reg_conf_mat= confusion_matrix(y_true=y_test,y_pred=predictions)

reg_accurasy_score=accuracy_score(y_test,predictions)


#arketa kalo apotelesma alla xeroume oti oi egrafes einai anises
reg_classification_report=classification_report(y_test,predictions)

#blepw oti to weighted avg tou f1 einai arketa kalo,idio me to accuracy opote katalabainw oti h daifora ton etiketwn den einai arketoi gai an allaxoun ta apotelesmata

#Kanw crossvalidation
kf = KFold( n_splits=5, shuffle=True)
list_conf_mat = []
for train_idx, test_idx in kf.split(X):
    cross_classifier = LogisticRegression(solver='liblinear')
    cross_classifier.fit(X[train_idx, :], y[train_idx])
    predictions = cross_classifier.predict(X[test_idx, :])
    list_conf_mat.append(confusion_matrix(y_true=y[test_idx],y_pred=predictions))
for conf_mat in list_conf_mat:
    print("Regression K fold :\n",conf_mat)
    print('')
k_fold_average_reg=np.mean([(cm[0, 0] + cm[1, 1]) / sum(sum(cm)) for cm in list_conf_mat])

#telos cross validationgia to logisticregression


#decision tree

tree = DecisionTreeClassifier(criterion='gini')
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
tree_conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions)
tree_accurasy_score=accuracy_score(y_test,predictions)
tree_classification_report=classification_report(y_test,predictions)
#k fold gia tree

kf = KFold( n_splits=5, shuffle=True)
list_conf_mat = []
for train_idx, test_idx in kf.split(X):
    tree = DecisionTreeClassifier()
    tree.fit(X[train_idx, :], y[train_idx])
    predictions = tree.predict(X[test_idx, :])
    list_conf_mat.append(confusion_matrix(y_true=y[test_idx],y_pred=predictions))
k_fold_average_tree=np.mean([(cm[0, 0] + cm[1, 1]) / sum(sum(cm)) for cm in list_conf_mat])
#telos
#k-nearest

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

neighbor_accurasy_score=clf.score(X_test,y_test)
predictions = clf.predict(X_test)
neighbor_conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions)
neighbor_classification_report=classification_report(y_test,predictions)
# k fold gia geitona

kf = KFold( n_splits=5, shuffle=True)
list_conf_mat = []
for train_idx, test_idx in kf.split(X):
    clf = DecisionTreeClassifier()
    clf.fit(X[train_idx, :], y[train_idx])
    predictions = clf.predict(X[test_idx, :])
    list_conf_mat.append(confusion_matrix(y_true=y[test_idx],y_pred=predictions))
k_fold_average_neighbor=np.mean([(cm[0, 0] + cm[1, 1]) / sum(sum(cm)) for cm in list_conf_mat])

#svc
clf2 = SVC(kernel='linear',C=3)
clf2.fit(X_train,y_train)
svm_accurasy_score=clf2.score(X_train,y_train)
predictions = clf2.predict(X_test)
svm_conf_mat = confusion_matrix(y_true=y_test, y_pred=predictions)
svm_classification_report=classification_report(y_test,predictions)
kf = KFold( n_splits=5, shuffle=True)
list_conf_mat = []
for train_idx, test_idx in kf.split(X):
    clf = DecisionTreeClassifier()
    clf.fit(X[train_idx, :], y[train_idx])
    predictions = clf.predict(X[test_idx, :])
    list_conf_mat.append(confusion_matrix(y_true=y[test_idx],y_pred=predictions))
k_fold_average_svm=np.mean([(cm[0, 0] + cm[1, 1]) / sum(sum(cm)) for cm in list_conf_mat])



#printing
print("Logistic regression confusion matrix:\n",reg_conf_mat)
print("Logistic regression accuracy score: ",reg_accurasy_score)
print("Logistic regression classification report:\n",reg_classification_report)
print("Logistic regression 5-fold cross validation average: ",k_fold_average_reg)
plot_confusion_matrix(logmodel,X_test,y_test)
plt.title('Logistic regression confusion matrix')
plt.show()


print("Decision tree confusion matrix:\n",tree_conf_mat)
print("Desicion tree accuracy score: ",tree_accurasy_score)
print("Decision tree classification report:\n",tree_classification_report)
print("Decision tree 5-fold cross validation average: ",k_fold_average_tree)
plot_confusion_matrix(tree,X_test,y_test)
plt.title('Decision tree confusion matrix')
plt.show()

print("K-nearest neighbor confusion matrix:\n",neighbor_conf_mat)
print("K-nearest neighbor accuracy score: ",neighbor_accurasy_score)
print("K-nearest neighbor classification report:\n",neighbor_classification_report)
print("K-nearest neighbor 5-fold cross validation average: ",k_fold_average_neighbor)
plot_confusion_matrix(clf,X_test,y_test)
plt.title('K-nearest neighbor confusion matrix')
plt.show()

print("Support vector machine confusion matrix:\n",svm_conf_mat)
print("Support vector machine accuracy score: ",svm_accurasy_score)
print("Support vector machine classification report:\n",svm_classification_report)
print("Support vector machine 5-fold cross validation average: ",k_fold_average_svm)
plot_confusion_matrix(clf2,X_test,y_test)
plt.title('Support vector machine confusion matrix')
plt.show()




ac_names=('Logistic Regression','Desicion Tree','K_neighbor','SVM')
ac_scores = (reg_accurasy_score,tree_accurasy_score,neighbor_accurasy_score,svm_accurasy_score)
y_pos = np.arange(len(ac_names))
plt.bar(y_pos,ac_scores,align='center',alpha=0.5,color=('r','b','g','black'))
plt.xticks(y_pos,ac_names)
plt.title('Classifiers Accuracy')
plt.show()

kfold_names=('Logistic Regression','Desicion Tree','K_neighbor','SVM')
kfold_scores = (k_fold_average_reg,k_fold_average_tree,k_fold_average_neighbor,k_fold_average_svm)
y_pos = np.arange(len(kfold_names))
plt.bar(y_pos,kfold_scores,align='center',alpha=0.5,color=('r','b','g','black'))
plt.xticks(y_pos,kfold_names)
plt.title('5-Fold Cross Validation Averages')
plt.show()
