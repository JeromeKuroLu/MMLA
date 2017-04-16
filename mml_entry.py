import Vectorizer_constrcutor
import pandas as pd

# full process
if __name__ == "__main__":
    # construct vectorizer
    constructor = Vectorizer_constrcutor.Vectorizer_constructor()
    constructor.construct_vectorizer_by_file('./data_set/test20170314060000.csv')
    vec_dirs = constructor.vec_dirs
    df = pd.read_csv('./data_set/test20170314060000.csv')
    validate_data = df.iloc[0:1, :]
    constructor.validate_vecs(vec_dirs, validate_data)
