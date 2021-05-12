# 无监督算法
import numpy as np

class Kmeans(object):

    def __init__(self, k):
        self.k = k


    def cal_dis(self,v1, v2):
        return np.sum(np.square(v1 - v2), axis=1).T

    def init_center(self, data):
        k          = self.k
        dimFeature = data.shape[1]
        # 这里一定要加np.mat 否则numpy的广播机制会出错
        # center[:,j].shape 是(3,)
        center     = np.mat(np.zeros((k, dimFeature)))
        for j in range(dimFeature):
            min_j = min(data[:, j])
            max_j = max(data[:, j])
            rangeJ = float(max_j - min_j)
            #print((min_j + rangeJ * np.random.rand(k, 1)).shape)
            #print(center[:,j].shape)
            center[:, j] = min_j + rangeJ * np.random.rand(k, 1)
        self.center = center
        #print("*****init center is ******",center)


    def do_work(self, data):
        k = self.k
        self.init_center(data)
        c = self.center
        c_change = True
        m = data.shape[0]
        # D 保存了 每个样本到每个群心的距离，最后一列为距离最小的群心
        D = np.mat(np.zeros((m, k)))
        iter = 0
        while c_change:
            dc = {}
            for y in range(k):
                dc.setdefault(y, [])

            for i in range(m):
                D[i, :]   = self.cal_dis(c, data[i, :])
                kk = np.argmax(D[i, :-1])
                dc[kk].append(data[i, :])
            iter += 1
            new_center = np.mat(np.zeros((k, 2)))
            for kkk, vvv in dc.items():
                for elem in dc[kkk]:
                    new_center[kkk,:] = new_center[kkk, :] + elem
                new_center[kkk, :] = new_center[kkk, :] / len(dc[kkk]) if len(dc[kkk]) != 0 else 0
            print(np.sum(self.cal_dis(c, new_center)))
            print(c)

            if np.sum(self.cal_dis(c, new_center)) < 0.1:
                c_change = True
            else:
                c = new_center
            if iter >= 6000:
                print("over MAX iteration  break!!!")
                break





if __name__ == "__main__":
    my_kmeans = Kmeans(3)
    data = np.array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1]])
    #my_kmeans.init_center(data)
    my_kmeans.do_work(data)