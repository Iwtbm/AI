import kaggle_submission as dt
import numpy as np
import time
import multiprocessing

class DecisionTreePart2Tests():
    """Tests for Decision Tree Learning.

    Attributes:
        restaurant (dict): represents restaurant data set.
        dataset (data): training data used in testing.
        train_features: training features from dataset.
        train_classes: training classes from dataset.
    """

    def setUp(self):
        my_data = np.genfromtxt('kaggle_train.csv', delimiter=',')
        self.classes = my_data[:,0].astype(int)
        self.features = my_data[:,1:]

        # self.dataset = dt.load_csv('kaggle_train.csv', class_index=0)
        # self.dataset = dt.load_csv('part23_data.csv')
        # self.dataset = dt.load_csv('challenge_train.csv', class_index=0)
        # self.features, self.classes = self.dataset

        self.process_num = 6
        self.n_estimators = 6    

    def test_decision_ChallengeClassifier(self):
        """Test decision tree classifies all data correctly.

        Asserts:
            classification is 100% correct.
        """
        m,n = self.features.shape
        pool = multiprocessing.Pool(self.process_num)
        test_data = np.genfromtxt('kaggle_test_unlabeled.csv', delimiter=',')

        start = time.time()
        result = []
        output = np.zeros(m)
        test_output = np.zeros(test_data.shape[0])

        for i in range(self.n_estimators):
            result.append(pool.apply_async(self.build_tree_mulprocess, args=(i, self.features, self.classes)))

        pool.close()
        pool.join()
        end = time.time()
        print('time:', (end - start))

        for r in result:
            tree = r.get()
            output += np.array(tree.classify(self.features))
            test_output += np.array(tree.classify(test_data))

        output = output/self.n_estimators - 0.5
        answers = list(np.int64(output>0))
        print('Accuracy test result: %f' % (dt.accuracy(answers, self.classes)))

        test_output = test_output/self.n_estimators - 0.5
        answers = list(np.int64(test_output>0))
        result_with_id = np.array([range(0,len(answers)), answers]).transpose()
        np.savetxt("kaggle_result.csv", result_with_id, fmt='%d', delimiter=",", header = "Id,Class")


    def build_tree_mulprocess(self, i, features, classes):
        """
        Building tree
        """
        print('Building tree ', i, ' ...')
        # sample_dataSet = self.get_sample_dataSet(dataSet, i)
        m, n = features.shape
        tree = dt.ChallengeClassifier(1, float('inf'), 1, 5) #int(np.round(np.sqrt(n)))
        tree.fit(features, classes)
        print('Build tree ', i, ' end')
        return tree

    # def org_decision_ChallengeClassifier(self):
    #     """Test decision tree classifies all data correctly.

    #     Asserts:
    #         classification is 100% correct.
    #     """
    #     dataset = dt.load_csv('challenge_train.csv', class_index=0)
    #     features, classes = dataset
    #     # print(classes.shape, features.shape)
    #     m,n = features.shape
    #     tree = dt.ChallengeClassifier(10, float('inf'), 1, int(np.round(np.sqrt(n))))
    #     t = time.time()
    #     tree.fit(features, classes)
    #     output = tree.classify(features)
    #     output = np.array(output)
    #     p = 1 - np.sum(abs(classes - output))/classes.shape[0]
    #     print(p)
    #     print(time.time()-t)

if __name__ == '__main__':
    d = DecisionTreePart2Tests()
    d.setUp()
    d.test_decision_ChallengeClassifier()
    # d.org_decision_ChallengeClassifier()