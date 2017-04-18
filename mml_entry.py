from Vectorizer import Vectorizer
from Dim_reduction import Dim_reduction
import pandas as pd
import dill
import numpy as np
from scipy.sparse import csr_matrix, vstack
import Classifier
import random
import multiprocessing

# full process
if __name__ == "__main__":
    # construct vectorizer
    constructor = Vectorizer()
    constructor.construct_vectorizer_by_file('./data_set/test20170314060000.csv')
    vec_dirs = constructor.vec_dirs

    df = pd.read_csv('./data_set/test20170314060000.csv')

    pool = multiprocessing.Pool(processes=4)
    vds = None
    i = 0
    step = 20000
    while i <= df.shape[0]:
        validate_data = df.iloc[i:(i+step), :]
        vd = pool.apply_async(constructor.validate_vecs, (vec_dirs, validate_data)).get()
        if i == 0:
            vds = vd
        else:
            vds = vstack([vds, vd])
        i = i + step
    pool.close()
    pool.join()

    idx = 0
    for vec_dir in vec_dirs:
        vec_name = vec_dir[vec_dir.rindex('vec_') + 4:vec_dir.rindex('.pkl')]
        if (vec_name != 'blType'):
            vec_model = dill.load(open(vec_dir, 'rb'))
            vec_dim_num = len(vec_model.get_feature_names())
            training_matrix = vds.tocsc()[:, idx:idx+vec_dim_num]
            m_pca = Dim_reduction().construct_pca_by_map(training_matrix.toarray())
            dill.dump(m_pca, open('./model/reduction/reduce_' + vec_name + '.pkl', 'wb'))
            idx += vec_dim_num

    dill.dump(vds, open('./data_set/vec_data_dill.pkl', 'wb'))

    # construct classifier
    # vec_file = open('./data_set/vec_data.pkl', 'rb')
    # tda = dill.load(vec_file).tocsc()
    # incremental_pca = IncrementalPCA(16)
    # tda_t = tda[:, 0:tda.shape[1] - 1]
    # t = 0
    # t_step = 10000
    # while t <= tda.shape[1]:
    #     tda_t_slice_data = tda[:, t:t+t_step].toarray()
    #     incremental_pca.partial_fit(tda_t_slice_data)
    # tda_t = incremental_pca.transform(tda_t)
    # tda_r = tda[:, -1:tda.shape[1]]
    # tda_r = np.reshape(tda_r.toarray(), tda.shape[0])
    # tda_r = tda_r.astype(int)
    #
    #
    # cls = Classifier.Classifier().construct_svm_classifier(tda_t, tda_r)
    # dill.dump(cls, open('./model/classifier/svm.pkl', 'wb'))


    # cls = dill.load(open('./model/classifier/svm.pkl', 'rb'))
    # test_row_num = 243
    # test_record = tda[test_row_num - 1:test_row_num + 100, 0:tda.shape[1] - 1]
    # actual_result = tda[test_row_num - 1:test_row_num + 100, -1:tda.shape[1]]
    # print(actual_result.toarray().T)
    # print(cls.predict(test_record).tolist())
