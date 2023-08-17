import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

def ohe(df, col):
    le = LabelEncoder()
    a = le.fit_transform(df[col]).reshape(-1, 1)
    columns = [ f'{col}_{name}' for name in le.classes_ ]
    encoder = OneHotEncoder(sparse=False, categories='auto')
    result = pd.DataFrame(encoder.fit_transform(a), columns=columns)
    return result

class NumInPartyTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        X['num_in_party'] = X['parch'] + X['sibsp']
        X['num_in_party'] = np.where(X['num_in_party'].isin([1]), 'Alone', 
                             np.where(X['num_in_party'].isin([2, 3, 4]), 'Small',
                             np.where(X['num_in_party'].isin([5, 6, 7]), 'Medium', 'Large')))
        result = ohe(X, 'num_in_party')
        return pd.concat([X, result], axis=1).drop(['parch', 'sibsp', 'num_in_party'], axis=1)
    
class AgeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, buckets=10):
        super().__init__()
        self.__buckets = buckets
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        X['age'] = X.groupby(['sex', 'pclass'])['age'].apply(lambda val : val.fillna(val.mean()))

        age_labels = [f'age_{x}' for x in range(self.__buckets) ]
        X['age_bucket'] = pd.qcut(X['age'], self.__buckets, labels=age_labels)
        age_buckets = ohe(X, 'age_bucket')
        return pd.concat([X, age_buckets], axis=1).drop(['age', 'age_bucket'],axis=1)
        
        
class CabinTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        X['deck'] = X['cabin'].apply(lambda cabin : cabin[0] if pd.notnull(cabin) else 'X')
        deck_encoder = LabelEncoder()
        X['deck'] = deck_encoder.fit_transform(X['deck'])
        return X.drop(['cabin'], axis=1)

class ColumnDropper(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
        self.__columns = columns
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return X.drop(self.__columns, axis=1)
    
class FareTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, buckets=10):
        super().__init__()
        self.__buckets = buckets
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        X['fare'] = X['fare'].fillna(0)

        fare_labels = [f'fare_{i}' for i in range(self.__buckets)]
        X['fare_bucket'] = pd.qcut(X['fare'], self.__buckets, labels=fare_labels)
        fare_buckets = ohe(X, 'fare_bucket')
        return pd.concat([X, fare_buckets], axis=1).drop(['fare', 'fare_bucket'],axis=1)
    
class EmbarkedTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        X['embarked'] = X['embarked'].fillna('S').astype('object')
        embarked_encoder = LabelEncoder()
        X['embarked'] = embarked_encoder.fit_transform(X['embarked'])
        return X
    
class SexTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        super().__init__()
        
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        le = LabelEncoder()
        X['sex'] = le.fit_transform(X['sex'])
        return X
    
data_pipe = Pipeline([
    ('num_in_party', NumInPartyTransformer()),
    ('age', AgeTransformer(7)),
    ('cabin', CabinTransformer()),
    ('fare', FareTransformer()),
    ('embarked', EmbarkedTransformer()),
    ('sex', SexTransformer()),
    ('cleanup', ColumnDropper(columns=['name', 'ticket', 'boat', 'home.dest', 'body', 'survived'])),
])