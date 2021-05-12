import numpy as np

class KNN(object):

    def __init__(self, k):
        self.k = k

    def cal_distance(self, train, test):
        print("distace array is ***", np.sum(np.square(train-test), axis=1))
        return np.sum(np.square(train-test), axis=1)

    def sort_distance(self, train, test):
        d = self.cal_distance(train, test)

        sort_index = np.argsort(d)
        top_k_index = sort_index[:self.k]
        print("The top k train data index is ", top_k_index)
        return top_k_index


    def vote(self, train, test, train_label):
        # top k vote
        v = {}
        top_k_index = self.sort_distance(train, test)
        for index in top_k_index:
            count = v.get(train_label[index], 0)
            v[train_label[index]] = count + 1
        return v

    # predict
    def do_work(self, train, test, train_lable):
        vote = self.vote(train, test, train_lable)
        predict_label = max(vote, key=vote.get)
        print("KNN predict result is ***",predict_label)
        return predict_label

if __name__ == "__main__":
    train_data  = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    train_lable = ['A', 'A', 'B', 'B']
    test_data   = np.array([1.2, 0.2])
    my_Knn = KNN(k=3)
    my_Knn.do_work(train_data, test_data, train_lable)