from sklearn.decomposition import IncrementalPCA
import numpy as np

class Dim_reduction():
    def construct_pca_by_map(self, training_data):
        dims = training_data.shape[1]
        m_pca = None
        if dims > 10:
            m_pca = IncrementalPCA(10)
        else:
            m_pca = IncrementalPCA(dims)
        if dims > 2000:
            idx = 0
            t_step = 1000
            while idx <= training_data.shape[0]:
                m_pca.partial_fit(training_data[idx:idx+t_step, :])
                idx += t_step
        else:
            m_pca.fit(training_data)
        return m_pca

