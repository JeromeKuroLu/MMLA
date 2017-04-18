from sklearn import svm

class Classifier:

    def construct_svm_classifier(self, training_data, training_label, clf=svm.SVC(decision_function_shape='ovo')):
        clf.fit(training_data, training_label)
        return clf