import numpy as np

import sys
sys.path.append("..")
import py_func.clustering as clustering

from scipy.cluster.hierarchy import linkage
import unittest

class TestClusteredSampling(unittest.TestCase):

    def test_Alg2(self):

        n, m = 10, 3

        # Importances/Weights for clients in federated problem
        P_uni = np.array([1 / n] * n)
        P_ratio = np.random.dirichlet(np.ones(n), size=1)[0]

        def test_Alg2_unit(sim_matrix: np.array, m: int, P: int):
            linkage_matrix = linkage(sim_matrix, "ward")

            clusters = clustering.get_clusters_with_alg2(linkage_matrix, m, P)
            P_new = np.mean(clusters, axis=0)
            norm = np.linalg.norm(P_new - P) # 求范数 默认是二范数
            self.assertAlmostEqual(norm, 0) # 是一个单元测试库函数，用于单元测试中以检查两个给定的值是否几乎相等。此函数将使用五个参数作为输入，并根据断言条件返回布尔值。

        # Test L1 or L2 similarity matrix
        sim_matrix = np.random.uniform(0, 10, (n, n)) # 在均匀分布中随机采样，左闭右开
        test_Alg2_unit(sim_matrix, m, P_uni) # 这个是上面定义出来的test_Alg2_unit函数
        test_Alg2_unit(sim_matrix, m, P_ratio)

        # Test cosine similarity matrix # 基于cosine相似矩阵
        sim_matrix = np.random.uniform(-1, 1, (n, n))
        test_Alg2_unit(sim_matrix, m, P_uni)
        test_Alg2_unit(sim_matrix, m, P_ratio)



if __name__ == '__main__':
    unittest.main()