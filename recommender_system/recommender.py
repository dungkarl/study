import pandas as pd 
import numpy as np 

u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=u_cols)
n_users = users.shape[0]

print('Number of users:', n_users)
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('ml-100k/ua.base', sep='\t', names=r_cols)
ratings_test = pd.read_csv('ml-100k/ua.test', sep='\t', names=r_cols)
rate_train = ratings_base.as_matrix()
rate_test = ratings_test.as_matrix()
#rate_test1 = ratings_test.get_values()
print('Number of traing rates:', rate_train.shape[0])
print('Number of test rates:', rate_test.shape[0])
# print(type(rate_test))
# print(type(rate_test1))
# print(rate_test.shape)
# print(rate_test1.shape)
# print(users)
# print(users['user_id'])

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL',
'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
#a = pd.read_csv('ml-100k/u.item', sep='|', encoding='raw_unicode_escape')
items = pd.read_csv('ml-100k/u.item', sep='|', names=i_cols, encoding='raw_unicode_escape')
n_items = items.shape[0]
print('Number of items:', n_items)
X0 = items.get_values()
X_train_counts = X0[:, -19:]
# print(rate_train[:4, :])
# print(items.head)
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=True, norm ='l2')
X = transformer.fit_transform(X_train_counts.tolist()).toarray()

def get_items_rated_by_user(rate_matrix, user_id):
    """
    return (item_ids, scores)
    """ 
    y = rate_matrix[:,0] # all users
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1
    # but id in python starts from 0
    ids = np.where(y == user_id +1)[0]
    item_ids = rate_matrix[ids, 1] - 1 # index starts from 0
    scores = rate_matrix[ids, 2]
    return (item_ids, scores)
    

from sklearn.linear_model import Ridge
from sklearn import linear_model
d = X.shape[1] # data dimension
W = np.zeros((d, n_users))
b = np.zeros(n_users)
for n in range(n_users):
    ids, scores = get_items_rated_by_user(rate_train, n)
    model = Ridge(alpha=0.01, fit_intercept = True)
    Xhat = X[ids, :]
    model.fit(Xhat, scores)
    W[:, n] = model.coef_
    b[n] = model.intercept_

#print(len(W[0]))
Yhat = X.dot(W) + b
n = 100
np.set_printoptions(precision=2) # 2 digits after .
ids, scores = get_items_rated_by_user(rate_test, 10)
print('Rated movies ids :', ids )
print('True ratings :', scores)
print('Predicted ratings:', Yhat[ids, n])



def evaluate(Yhat, rates, W, b):
    se = cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred
        se += (e*e).sum(axis = 0)
        cnt += e.size
    return np.sqrt(se/cnt)
print('RMSE for training: %.2f' %evaluate(Yhat, rate_train, W, b))
print('RMSE for test : %.2f' %evaluate(Yhat, rate_test, W, b))