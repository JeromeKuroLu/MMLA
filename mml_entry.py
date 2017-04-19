from Vectorizer import Vectorizer
from Dim_reduction import Dim_reduction
import pandas as pd
import pickle
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

    # pool = multiprocessing.Pool(processes=4)
    # vds = None
    # i = 0
    # step = 20000
    # while i <= df.shape[0]:
    #     validate_data = df.iloc[i:(i+step), :]
    #     vd = pool.apply_async(constructor.validate_vecs, (vec_dirs, validate_data)).get()
    #     if i == 0:
    #         vds = vd
    #     else:
    #         vds = vstack([vds, vd])
    #     i = i + step
    # pool.close()
    # pool.join()
    #
    # idx = 0
    # for vec_dir in vec_dirs:
    #     vec_name = vec_dir[vec_dir.rindex('vec_') + 4:vec_dir.rindex('.pkl')]
    #     if (vec_name != 'blType'):
    #         vec_model = dill.load(open(vec_dir, 'rb'))
    #         vec_dim_num = len(vec_model.get_feature_names())
    #         training_matrix = vds.tocsc()[:, idx:idx+vec_dim_num]
    #         m_pca = Dim_reduction().construct_pca(training_matrix.toarray())
    #         dill.dump(m_pca, open('./model/reduction/reduce_' + vec_name + '.pkl', 'wb'))
    #         idx += vec_dim_num
    #
    # dill.dump(vds, open('./data_set/vec_data_dill.pkl', 'wb'))

    # construct classifier
    vec_file = open('./data_set/vec_data.pkl', 'rb')
    tda = dill.load(vec_file).tocsc()
    vec_file.close()
    # tda_t = tda[:, 0:tda.shape[1] - 1]
    # tda_ar = None
    # p = 0
    # for col_name in df.columns:
    #     if col_name != 'blType':
    #         col_rdc = dill.load(open('./model/reduction/reduce_' + col_name + '.pkl', 'rb'))
    #         vec_model = dill.load(open('./model/vectorizer/vec_' + col_name + '.pkl', 'rb'))
    #         vec_dim_num = len(vec_model.get_feature_names())
    #         print('dimensionality reduction for ' + col_name + ', dim num is ' + str(vec_dim_num))
    #         col_matrix = tda_t[:, p:p + vec_dim_num]
    #         x = col_rdc.transform(col_matrix.toarray())
    #         if tda_ar is None:
    #             tda_ar = x
    #         else:
    #             tda_ar = np.concatenate((tda_ar, x), axis=1)
    #         p += vec_dim_num
    # dill.dump(tda_ar, open('./data_set/tda_ar.pkl', 'wb'))
    # w_pca = Dim_reduction().construct_pca(tda_ar)
    # dill.dump(w_pca, open('./model/reduction/reduce_all.pkl', 'wb'))
    # tda_nt = w_pca.transform(tda_ar)
    tda_nt = dill.load(open('./data_set/tda_nt.pkl', 'rb'))
    # dill.dump(tda_nt, open('./data_set/tda_nt.pkl', 'wb'))
    # tda_r = tda[:, -1:tda.shape[1]]
    # tda_r = np.reshape(tda_r.toarray(), tda.shape[0])
    # tda_r = tda_r.astype(int)
    #
    # print('build classifier')
    # cls = Classifier.Classifier().construct_svm_classifier(tda_nt, tda_r)
    # print('classifier builds successfully.')
    # dill.dump(cls, open('./model/classifier/svm.pkl', 'wb'))
    # print('dill for svm classifier')

    cls = dill.load(open('./model/classifier/svm.pkl', 'rb'))
    test_row_num = 0
    test_record = tda_nt[test_row_num:test_row_num + 67185, :]
    actual_result = tda[test_row_num:test_row_num + 67185, -1:tda.shape[1]].toarray()
    print(actual_result.T)
    pre_result = cls.predict(test_record)
    print(pre_result.tolist())
    same_num = 0
    for i in range(pre_result.shape[0]):
        pre_item = pre_result[i]
        actual_item = actual_result[i, 0]
        if pre_item == actual_item:
            same_num += 1
    print(same_num/67185)
