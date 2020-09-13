import sklearn.model_selection
import sklearn.svm
from collections import defaultdict
import numpy as np
import operator
import os


class Model:
    def __init__(self):
        self.processed_data = defaultdict(lambda: [])
        self.best_c = None


    def load_data(self, data_file, save_file_path=None):
        '''
        Load data and splits data and labels into dictionaries.
        '''
        season_index_dict = {
            9: "Spring",
            10: "Summer",
            11: "Autumn",
            12: "Winter"
        }

        f = open(data_file, "r")
        data = f.readlines()

        for sample in data:
            seasons = []
            sample = sample.strip('\n')
            sample = sample.split(' ')
            img = sample[0]
            annotations = sample[1:]
            indexes = [i for i,val in enumerate(annotations) if val=='1']
            
            if 9 in indexes:
                seasons.append(9)
            if 10 in indexes:
                seasons.append(10)
            if 11 in indexes:
                seasons.append(11)
            if 12 in indexes:
                seasons.append(12)
            if len(seasons) == 1:
                self.processed_data[season_index_dict[seasons[0]]].append(img)

        self.train = defaultdict(lambda: [])
        self.val = defaultdict(lambda: [])
        self.test = defaultdict(lambda: [])

        if save_file_path is not None:
            try:
                os.makedirs(save_file_path)
            except:
                pass
            save_dict = defaultdict(lambda: defaultdict(lambda: []))

        for k in self.processed_data.keys():
            # NOTE: no shuffling of data during splitting
            data = sklearn.model_selection.train_test_split(self.processed_data[k], train_size=0.6, shuffle=False)
            temp_train = data[0]
            data = data[1]
            data = sklearn.model_selection.train_test_split(data, train_size=0.375, shuffle=False)
            temp_val = data[0]
            temp_test = data[1]

            if save_file_path is not None:
                save_dict['train'][k] = np.asarray(temp_train)
                save_dict['val'][k] = np.asarray(temp_val)
                save_dict['test'][k] = np.asarray(temp_test)

            self.train['data'].extend(temp_train)
            labels = [k] * len(temp_train)
            self.train['labels'].extend(labels)

            self.val['data'].extend(temp_val)
            labels = [k] * len(temp_val)
            self.val['labels'].extend(labels)

            self.test['data'].extend(temp_test)
            labels = [k] * len(temp_test)
            self.test['labels'].extend(labels)

        if save_file_path is not None:
            for k in save_dict.keys():
                for j in save_dict[k].keys():
                    np.save(save_file_path + str(k) + '_' + str(j), save_dict[k][j])


    def train_for_c(self, features_path):
        '''
        Train four SVMs and test on validation set to get best C value.
        '''
        c_list = [0.01, 0.1, 0.1**0.5 , 1, 10**0.5 , 10, 1000**0.5]
        class_wise_accuracy_val = {}

        for c in c_list:
            # get class-wise averaged accuracy
            class_wise_accuracy_val[c] = self.get_results(features_path, c=c)

        self.best_c = max(class_wise_accuracy_val.items(), key=operator.itemgetter(1))[0]
        print("\nC value class-wise accuracies for validation set: " + str(class_wise_accuracy_val))


    def get_results(self, features_path, c, test_type='val', acc_type='vanilla'):
        '''
        Get performance measures for validation or test datasets.
        '''
        svm_index = {
            0: "Spring",
            1: "Summer",
            2: "Autumn",
            3: "Winter"
        }

        self.spring_svm = sklearn.svm.LinearSVC(C=c)
        self.summer_svm = sklearn.svm.LinearSVC(C=c)
        self.autumn_svm = sklearn.svm.LinearSVC(C=c)
        self.winter_svm = sklearn.svm.LinearSVC(C=c)

        length = 0

        if test_type == 'val':
            self.spring_train_indexes = [i for i,val in enumerate(self.train['labels']) if val=='Spring']
            self.summer_train_indexes = [i for i,val in enumerate(self.train['labels']) if val=='Summer']
            self.autumn_train_indexes = [i for i,val in enumerate(self.train['labels']) if val=='Autumn']
            self.winter_train_indexes = [i for i,val in enumerate(self.train['labels']) if val=='Winter']
            X = []
            for name in self.train['data']:
                X.append(np.load(features_path + name + '_ft.npy'))
            length = len(self.spring_train_indexes) + len(self.summer_train_indexes) + len(self.autumn_train_indexes) + len(self.winter_train_indexes)

            X_test = []
            for name in self.val['data']:
                X_test.append(np.load(features_path + name + '_ft.npy'))

        # train on both train and validation data
        elif test_type == 'test':
            self.spring_train_indexes = [i for i,val in enumerate(self.train['labels']) if val=='Spring'] + [i for i,val in enumerate(self.val['labels']) if val=='Spring']
            self.summer_train_indexes = [i for i,val in enumerate(self.train['labels']) if val=='Summer'] + [i for i,val in enumerate(self.val['labels']) if val=='Summer']
            self.autumn_train_indexes = [i for i,val in enumerate(self.train['labels']) if val=='Autumn'] + [i for i,val in enumerate(self.val['labels']) if val=='Autumn']
            self.winter_train_indexes = [i for i,val in enumerate(self.train['labels']) if val=='Winter'] + [i for i,val in enumerate(self.val['labels']) if val=='Winter']
            X = []
            for name in self.train['data']:
                X.append(np.load(features_path + name + '_ft.npy'))
            for name in self.val['data']:
                X.append(np.load(features_path + name + '_ft.npy'))

            X_test = []
            for name in self.test['data']:
                X_test.append(np.load(features_path + name + '_ft.npy'))

        length = len(self.spring_train_indexes) + len(self.summer_train_indexes) + len(self.autumn_train_indexes) + len(self.winter_train_indexes)

        y = [0] * length
        # train spring SVM
        for ind in self.spring_train_indexes:
            y[ind] = 1
        self.spring_svm.fit(X, y)

        # train summer SVM
        y = [0] * length
        for ind in self.summer_train_indexes:
            y[ind] = 1
        self.summer_svm.fit(X, y)

        # train autumn SVM
        y = [0] * length
        for ind in self.autumn_train_indexes:
            y[ind] = 1
        self.autumn_svm.fit(X, y)

        # train winter SVM
        y = [0] * length
        for ind in self.winter_train_indexes:
            y[ind] = 1
        self.winter_svm.fit(X, y)

        # testing with w.x + b
        spring_pred = [np.dot(self.spring_svm.coef_, x) + self.spring_svm.intercept_ for x in X_test]
        summer_pred = [np.dot(self.summer_svm.coef_, x) + self.summer_svm.intercept_ for x in X_test]
        autumn_pred = [np.dot(self.autumn_svm.coef_, x) + self.autumn_svm.intercept_ for x in X_test]
        winter_pred = [np.dot(self.winter_svm.coef_, x) + self.winter_svm.intercept_ for x in X_test]
        
        final_pred = []
        for i in range(len(X_test)):
            check = [spring_pred[i],summer_pred[i],autumn_pred[i],winter_pred[i]]
            best_index = check.index(max(check))
            final_pred.append(svm_index[best_index])

        if test_type == 'val':
            return self.get_accuracy(self.val['labels'], final_pred, classes=['Spring','Summer','Autumn','Winter'], acc_type=acc_type)
        elif test_type == 'test':
            return self.get_accuracy(self.test['labels'], final_pred, classes=['Spring','Summer','Autumn','Winter'], acc_type=acc_type)
        else:
            return


    def get_accuracy(self, y, pred, classes=None, acc_type='vanilla'):
        '''
        Get accuracy performance measures.
        '''
        assert len(y) == len(pred)

        if acc_type == 'vanilla':
            cnt = 0
            length = len(y)
            for i in range(length):
                if y[i] == pred[i]:
                    cnt += 1
            return cnt / length

        elif acc_type == 'class':
            assert len(classes) > 1

            num_classes = len(classes)
            cnt_dict = defaultdict(lambda: 0)
            length_dict = defaultdict(lambda: 0)

            for c in classes:
                length_dict[c] = y.count(c)

                for i in range(len(y)):
                    if y[i] == c:
                        if pred[i] == c:
                            cnt_dict[c] += 1
            
            acc = 0
            for c in classes:
                acc += cnt_dict[c] / length_dict[c]
            
            return acc / num_classes


# TODO: Change paths for relevant file(s), with a slash at the back for directories
features_path = './imagecleffeats/imageclef2011_feats/'
annotations_path = './trainset_gt_annotations.txt'
saved_train_val_test_splits_dir = './train_val_test_splits/'

model = Model()
model.load_data(annotations_path, save_file_path=saved_train_val_test_splits_dir)
model.train_for_c(features_path)

print("\nBest C value is: " + str(model.best_c))

results = {}
results['val_vanilla'] = model.get_results(features_path=features_path, c=model.best_c, test_type='val', acc_type='vanilla')
results['test_vanilla'] = model.get_results(features_path=features_path, c=model.best_c, test_type='test', acc_type='vanilla')
results['val_class'] = model.get_results(features_path=features_path, c=model.best_c, test_type='val', acc_type='class')
results['test_class'] = model.get_results(features_path=features_path, c=model.best_c, test_type='test', acc_type='class')

print("\nAccuracy results: " + str(results))