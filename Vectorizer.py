import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from functools import reduce
import pickle
import os
import numpy as np
from scipy.sparse import csr_matrix, hstack

class Vectorizer:
    def __init__(self):
        self.vec_prefix = './model/vectorizer/vec_'
        self.file_suffix = '.pkl'
        self.vec_dirs = []

    def construct_vectorizer_by_file(self, file_dir):
        df = pd.read_csv(file_dir)
        for col_name in df.columns:
            vec_file_dir = self.vec_prefix + col_name + self.file_suffix
            if(vec_file_dir not in self.vec_dirs):
                self.vec_dirs.append(vec_file_dir)
            vec_file_exists = os.path.exists(vec_file_dir)
            if(not vec_file_exists and col_name != 'blType'):
                print('generate vec for: ' + col_name)
                idx = df.columns.tolist().index(col_name)
                column_value = df.iloc[:, idx]
                column_value_list = column_value.tolist()
                func = lambda x, y: x if y in x else x + [y]
                distinct_column_value_list = reduce(func, [[], ] + column_value_list)
                # fit vectorizer
                training_data = []
                for el in distinct_column_value_list:
                    r = {column_value.name: str(el)}
                    training_data.append(r)
                vec = DictVectorizer()
                vec.fit(training_data)
                # persist vectorizer
                vec_file = open(vec_file_dir, 'wb')
                pickle.dump(vec, vec_file)

    def validate_vecs(self, vec_dirs, validate_data = None):
        result = None
        if(isinstance(vec_dirs, list)):
            for vec_dir in vec_dirs:
                vec_model = None
                vec_name = vec_dir[vec_dir.rindex('vec_') + 4:vec_dir.rindex('.pkl')]
                if(vec_name != 'blType'):
                    vec_model = pickle.load(open(vec_dir, 'rb'))
                validate_data_s = validate_data.loc[:, vec_name]
                r = self.validate_vec(vec_model, vec_name, validate_data_s)
                if(result is None):
                    result = r
                else:
                    result = hstack([result.tocoo(), r.tocoo()])
        return result

    def validate_vec(self, vec_model, vec_name, validate_data_s):
        td = []
        bl_type_values = []
        if(vec_name != 'blType'):
            validate_list = []
            for d in validate_data_s.tolist():
                e = {vec_name: str(d)}
                validate_list.append(e)
            td = vec_model.transform(validate_list)
        else:
            for d in validate_data_s.tolist():
                if(d not in bl_type_values):
                    td.append([len(bl_type_values)])
                    bl_type_values.append(d)
                else:
                    td.append([bl_type_values.index(d)])
            td = csr_matrix(td)
        return td