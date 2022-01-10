import pandas as pd
import numpy as np
import os
from pathlib import Path

path = Path(os.getcwd())
base_dir = path.parent
data_in = os.path.join(str(base_dir),"datasets/")
#for future possibly find way to pull data into our dataset folder through a separate file
hotels_stata, metastata = pd.read_csv(os.path.join(data_in,"hotels-europe_price.dta" ), encoding="latin1")
prices_stata, metastata = pd.read_csv(os.path.join(data_in,"hotels-europe_features.dta" ), encoding="latin1")

#[]
#
hotels_stata.info()

#[]
#
prices_stata.info()

#[]
#
hotels_stata.hotel_id.nunique()

#[]
prices_stata.hotel_id.nunique()

#[]
hotels_europe = hotels_stata.merge(prices_stata,left_on='hotel_id', right_on='hotel_id')
hotels_europe.info()

#[]
#
hotels_europe.head()

#[]
#
import matplotlib.pyplot as plt
%matplotlib inline

hotels_europe.hist(bins=30,figsize=(15,20))
# a lot of the values are skewed in future could take log transformations of some of them but
# specifically the price which we are trying to predict

#[]
#
hotels_europe.country.value_counts()

#[]
#
#only really want data for a specific country
# try dropping values only after having specififed the country
hotels_europe_france = hotels_europe.loc[hotels_europe.country=='France',:]

#[]
#
hotels_europe_france.info()

#[]
#
hotels_europe_france.distance.value_counts()

#[]
#
hotels_europe_france.distance.value_counts().sort_index()

#[]
#

hotels_europe_france.distance.value_counts(bins=5).sort_index()

#[]
#

#use this method to categorize the data and split accordingly
#must see if it it's better to define the bins or just use defaults
hotels_europe_france.loc[:,'distance_cat'] = pd.cut(hotels_europe_france.distance, bins=[-np.inf,4,8,13,17.,np.inf], labels=[1,2,3,4,5]) # np.inf and -np.inf is important here and 0.0
hotels_europe_france['distance_cat'].value_counts()

#[]
#
hotels_europe_france['distance_cat'].value_counts(dropna=False,normalize=True)

#[]
#

hotels_europe_france.reset_index(drop=True,inplace=True)
hotels_europe_france.shape

#[]
#
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.20, train_size=0.80,random_state=42)

for train_index, test_index in split.split(hotels_europe_france,hotels_europe_france['distance_cat']):
        train_set = hotels_europe_france.loc[train_index,:]
        test_set = hotels_europe_france.loc[test_index,:]

#[]
#

train_set.shape

#[]
#
test_set.shape

#[]
#
train_set.distance_cat.value_counts(dropna=False,normalize=True)

#[]
#
test_set.distance_cat.value_counts(dropna=False,normalize=True)

#[]
# can start exploring more now
for a in [train_set,test_set]:
    a.drop(columns = ['distance_cat'],inplace=True)
    a.reset_index(drop=True,inplace=True)

train_set.head()


#[]
#
train_set.info()

#[]
#
hotels = train_set.copy()

#[]
#

for i in ['ratingta_count','rating_count']: #converting to float to be used asa numerical column
    hotels.loc[:,i] = hotels.loc[:,i].astype(float)

#[]
#
hotels.info()

#[]
#
from pandas.plotting import scatter_matrix
# here we can see the correlation between different numeric values
scatter_matrix(hotels[['price','weekend','holiday','distance','stars','rating']], figsize=(15,20))

#[]
#

hotels.corr()['price'].sort_values(ascending=False)
#[]
#
# exploring the values of nnights and their effect on price
hotels[['nnights','price']].sort_values('price')

#{]
#
hotels[['nnights','price']].sort_values(['nnights','price'],ascending=False)

#[]
#
hotels.price.value_counts(bins=4).sort_index(ascending=True) #the price distribution before accounting for nnights

(hotels.price / hotels.nnights).value_counts(bins=4).sort_index(ascending=True) #the price distribution after accounting for nnights_counts(bins=4).sort_index(ascending=True)
#[]
#
hotels.loc[:,'price_one_night'] = hotels.price / hotels.nnights

# * seeing if transforming price by nnight variable makes a difference and it does a lot at least for \
# the most important variables
df1 = pd.DataFrame(hotels.corr()['price_one_night'])
df1 = pd.concat([df1,hotels.corr()['price']], axis=1)
df1.sort_values('price_one_night')

#[]
#
#[]
#
scatter_matrix(hotels[['price_one_night','distance','stars','ratingta']], figsize=(15,20))



# * checking out some cat. variables \
# in the future stars could be considered a categorical variable

#[]
#
hotels.select_dtypes(include='object')

#[]
#

hotels.accommodation_type.value_counts()

#[]
#
hotels.offer_cat.value_counts()

#[]
#
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
acc_cat = cat_encoder.fit_transform(hotels[['accommodation_type','offer_cat']])

#[]
#
print(acc_cat.toarray())
print(acc_cat.toarray().shape)

#[]
#
cat_encoder.categories_

# the variables we will seek to keep are
# * distance
# * ['price_one_night','distance','stars','ratingta']
# * ['accommodation_type','offer_cat']

#[]
#

hotels.corr()['ratingta'].sort_values()
hotels.corr()['stars'].sort_values()

#[]
#

impute_col = ['stars','ratingta','rating','price_one_night']
index_sample = hotels.loc[(hotels['ratingta'].isna()) & (hotels['stars'].isna()), impute_col].head(10).index
hotels.loc[(hotels['ratingta'].isna()) & (hotels['stars'].isna()), impute_col].head(10)
#some sample values for reference before doing the transformation
#[]
#
hotels.loc[:,impute_col].agg(['mean','median'])
#some aggregations for reference before doing the transformation
#[]
#
hotels_to_impute = hotels.loc[:,impute_col].copy()
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=7)
#[]
#
hotels_to_impute_new_values = imputer.fit_transform(hotels_to_impute)

#[]
#
hotels_tr = pd.DataFrame(hotels_to_impute_new_values, columns = hotels_to_impute.columns, index=hotels_to_impute.index)
hotels_tr.iloc[index_sample,:]

#[]
#
hotels_tr.loc[:,impute_col].agg(['mean','median'])

#[]
#




from sklearn.base import BaseEstimator,TransformerMixin
#[]
#
num_col = ['ratingta_count','rating_count','nnights','price']

ratingta_count_ix, rating_count_ix, nnights_ix, price_ix = [hotels.columns.get_loc(x) for x in num_col]
#[]
#
print(hotels.values[:,price_ix].sum()) # reference to make sure values are changed
print((hotels.values[:,price_ix] / hotels.values[:,nnights_ix]).sum())
#[]
#
class hotels_attribute_adder(BaseEstimator,TransformerMixin):

    def __init__(self,change_rating_count_float=False):
        self.change_rating_count_float = change_rating_count_float

    def fit(self,X,y=None):
        return self #nothing to fit since we are just creating new and editing new columns
    def transform(self,X):

        ratingta_count_float = X[:, ratingta_count_ix].astype(float)
        rating_count_float = X[:, rating_count_ix].astype(float)
        price_one_night = X[:,price_ix] / X[:,nnights_ix]

        if self.change_rating_count_float :
                array = np.c_[X, ratingta_count_float, rating_count_float, price_one_night]
                array = np.delete(array,[ratingta_count_ix,rating_count_ix] , axis=1)
                return array
        else:
                return np.c_[X, price_one_night]


#[]
#
attr_adder = hotels_attribute_adder(change_rating_count_float=True)
attr_adder
#[]
#
hotels_test = attr_adder.transform(hotels.values)
#[]
#
hotels_test.shape

#[]
#
hotels_extra_attr = pd.DataFrame(hotels_test, columns = hotels.columns.tolist() + ['ratingta_count_float'] +\
        ['rating_count_float'] + ['price_one_night'], index=hotels.index)
hotels_extra_attr.head()

hotels_extra_attr.agg(['mean','median'])

#[]
#
hotels.loc[:,impute_col].agg(['mean','median']) #clean transformation

#[]
#

from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([('attr_add', hotels_attribute_adder(change_rating_count_float=True)),\
                         ('imputer', KNNImputer(n_neighbors=7))]) # this pipeline forces me to use .values
num_pipeline2 = Pipeline(
                        [('imputer', KNNImputer(n_neighbors=7)),
                        ('attr_add', hotels_attribute_adder(change_rating_count_float=True)) #this pipeline throws an array into our transformer
                        ])

#[]
#
pipeline_col = ['hotel_id',
 'price',
 'scarce_room',
 'offer',
 'year',
 'month',
 'weekend',
 'holiday',
 'nnights',
 'rating_count',
 'distance',
 'distance_alter',
 'stars',
 'ratingta',
 'ratingta_count',
 'rating',
 'price_one_night','ratingta_count','rating_count']

hotels_num = hotels.loc[:,pipeline_col]
hotels_num.shape
#[]
#

num_col = ['ratingta_count','rating_count','nnights','price']
ratingta_count_ix, rating_count_ix, nnights_ix, price_ix = [hotels_num.columns.get_loc(x) for x in num_col]
#[]
#
hotel_num_transform = num_pipeline2.fit_transform(hotels_num)

hotel_num_transform.shape

hotel_num_transform
#[]
#
hotels_extra_attr2 = pd.DataFrame(hotel_num_transform, columns = hotels_num.columns.tolist() + ['ratingta_count_float'] +\
        ['rating_count_float'] + ['price_one_night'], index=hotels_num.index)
#[]
#

hotels_extra_attr2
#[]
#
num_col = ['ratingta_count','rating_count','nnights','price']

ratingta_count_ix, rating_count_ix, nnights_ix, price_ix = [x for x in enumerate(num_col)]
cat_attribs  = ['accommodation_type','offer_cat']
num_attribs = ['ratingta_count','rating_count','nnights','price','stars','ratingta','rating']
from sklearn.compose import ColumnTransformer

full_pipeline = ColumnTransformer([
                        ('num_val', num_pipeline2, num_attribs), #in the future separate KNNimputer so that I can use 'price_on_night' to imputes values as well
                        ('cat_val', OneHotEncoder(), cat_attribs)])

hotels1_prepared = full_pipeline.fit_transform(hotels1)

#[]
#


#[]
#

hotels1_prepared.info()

hotels1_prepared.agg(['mean','median'])





#[]
#

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(hotels1_prepared_cut,y_labels)

#[]
#
#testing on some samples

sample_data = hotels1.iloc[:5,:]
prepared_sample_data = full_pipeline.transform(sample_data)
sample_y_labels = prepared_sample_data[:,6]
prepared_sample_data = np.delete(prepared_sample_data,list(range(7)), axis=1)


#[]
#
print('Predictions:', lin_reg.predict(prepared_sample_data), sep='\n')

#[]
#
#actual values
print('actual values:', sample_y_labels, sep='\n')

#[]
#
hotel_predictions = lin_reg.predict(hotels1_prepared_cut)

from sklearn.metrics import mean_squared_error
lin_rmse = mean_squared_error(y_labels,hotel_predictions,squared=False)

#[]
#


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(y_labels, hotel_predictions)
lin_mae

#[]
#
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(hotels1_prepared_cut,y_labels)
tree_hotel_predictions = tree_reg.predict(hotels1_prepared_cut)
tree_rmse = mean_squared_error(y_labels,tree_hotel_predictions, squared=False)
tree_rmse

#[]
#
from sklearn.model_selection import cross_val_score

tree_scores = cross_val_score(tree_rmse, hotels1_prepared_cut, y_labels, cv=5, scoring='neg_mean_squared_error')
tree_scores_rmse = np.sqrt(-tree_scores)

def show_scores(scores):
        print('scores:',scores)
        print('mean:',scores.mean())
        print('std:',scores.std())

#[]
#
show_scores(tree_scores_rmse)

#[]
#
lin_reg_scores = cross_val_score(lin_reg,hotels1_prepared_cut,y_labels,cv=5,scoring='neg_mean_squared_error')
lin_reg_scores_rmse = np.sqrt(-lin_reg_scores)
show_scores(lin_reg_scores_rmse)

#[]
#
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100,random_state=42)
forest_reg.fit(hotels1_prepared_cut,y_labels)
forest_hotel_predictions = forest_reg.predict(hotels1_prepared_cut)
forest_rmse = mean_squared_error(y_labels,forest_hotel_predictions,squared=False)
forest_rmse

#[]
#
forest_scores = cross_val_score(forest_reg,hotels1_prepared_cut,y_labels,cv=5,scoring='neg_mean_squared_error')
forest_scores_rmse = np.sqrt(-forest_scores)
show_scores(forest_scores_rmse)

#[]
#
from sklearn.svm import SVR
svm_reg = SVR(kernal='linear')
svm_reg.fit(hotels1_prepared_cut,y_labels)
svm_predictions = svm_reg.predict(hotels1_prepared_cut)
svm_rmse = mean_squared_error(y_labels,svm_predictions,squared=False)
svm_rmse

#[]
#
from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators':[10,25,50], 'random_state':42, 'max_features':[2,4,6,8]}, \
                {bootstrap:[False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]

forest_reg = RandomForestRegressor(random_state=42)
clf = GridSearchCV(forest_reg, param_grid, return_train_score=True, scoring='neg_mean_squared_error', cv=5 )
clf.fit(hotels1_prepared_cut,y_labels)

#[]
#
clf.best_params_ # best hyperparameter combination

#[]
#
clf.best_estimator_

#[]
#

clf.cv_results_
cv_scores = clf.cv_results_
for mean_score, params in zip(cv_scores['mean_test_score'], cv_scores['params']):
        print(np.sqrt(-mean_score), params)

#[]
#
pd.DataFrame(clf.cv_results_)

#[]
#
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_distribution = {'n_estimators':randint(low=1,high=100), 'max_features':randint(low=1,high=8)}
forest_reg = RandomForestRegressor(random_state=42)
random_search_cv = RandomizedSearchCV(forest_reg, param_distribution, scoring='neg_mean_squared_error', return_train_score=True, cv=5 )

#[]
#
random_search_cv.fit(hotels1_prepared_cut,y_labels)
random_search_cv.cv_results_

#[]
#
cv_scores = random_search_cv.cv_results_
for mean_score , params in zip(cv_scores['mean_test_score'], cv_scores['params']):
        print(np.sqrt(-mean_score), params)

#[]
#
random_search_cv.best_estimator_.feature_importances_

#[]
#
full_pipeline.named_transformers_

#[]
#
full_pipeline.named_transformers_['num_val'].categories_
accomodation_cat = full_pipeline.named_transformers_['cat_val'].categories_[0].tolist()
offer_cat = full_pipeline.named_transformers_['cat_val'].categories_[1].tolist()
manual_columns = ['stars','ratingta','rating','distance']
#[]
#
features = random_search_cv.best_estimator_.feature_importances_
labels = manual_columns + accomodation_cat + offer_cat
for feature, label in zip(features,labels):
        print(feature,label)

#[]
#
sorted(zip(features,labels),reverse=True)

#[]
#
forest_reg_final = clf.best_estimator_

#[]
#
hotels_test_prepared = full_pipeline.transform(test_set)

y_test_labels = hotels_test_prepared[:,6]
hotels_test_prepared_cut = np.delete[hotels_test_prepared,list(range(7)),axis=1]

#[]
#
forest_reg_final_predictions = forest_reg_final.predict(hotels_test_prepared_cut)

#[]
#
forest_reg_final_rmse = mean_squared_error(y_test_labels, forest_reg_final_predictions, squared=False)

#[]
#
from scipy import stats

confidence = 0.95
squared_errors = (forest_reg_final_predictions - y_test_labels) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
#[]
#
import joblib




# *****
 #had i dropped null values at beginnning

# *****
#[]
#
# just dropping these rows now but in future will attempt imputation
# come back and try to use KNN to impute a value for these columns
# check if the rows were dropping share any particular characteristics
hotels_europe_no_na = hotels_europe.dropna(subset=['rating_count','stars', 'ratingta', 'ratingta_count', 'rating'])
hotels_europe_no_na.info()

#[]
#

import matplotlib.pyplot as plt
%matplotlib inline

hotels_europe_no_na.hist(bins=30,figsize=(15,20))
#[]
#
#a lot of the values are skewed in future could take log transformations of some of them but specifically the price which we are trying to predict

#[]
#
#only really want data for a specific country
# try dropping values only after having specififed the country
hotels_europe_no_na.country.value_counts()
hotels_europe_no_na_france = hotels_europe_no_na[hotels_europe_no_na.country=='France']

#[]
#

#going to split data now to prevent risk of data-snooping bias
#there are many different values in distance so we'll create bins for it
hotels_europe_no_na_france.distance.value_counts()
hotels_europe_no_na_france.distance.value_counts().sort_index()

#[]
#
#use this method to categorize the data and split accordingly
#must see if it it's better to define the bins or just use defaults
hotels_europe_no_na_france.loc[:,'distance_cat'] = pd.cut(hotels_europe_no_na_france.distance, bins=[-np.inf,4,8,13,17.,np.inf], labels=[1,2,3,4,5]) # np.inf and -np.inf is important here and 0.0
hotels_europe_no_na_france['distance_cat'].value_counts()

#[]
#
hotels_europe_no_na_france['distance_cat'].value_counts(dropna=False,normalize=True)

#[]
#
hotels_europe_no_na_france.reset_index(drop=True,inplace=True)
hotels_europe_no_na_france.shape
#[]
#

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.20, train_size=0.80,random_state=42)

for train_index, test_index in split.split(hotels_europe_no_na_france,hotels_europe_no_na_france['distance_cat']):
        train_set = hotels_europe_no_na_france.loc[train_index,:]
        test_set = hotels_europe_no_na_france.loc[test_index,:]

#[]
#
train_set.shape
#[]
#
test_set.shape

#[]
#
train_set.distance_cat.value_counts(dropna=False,normalize=True)

#[]
#
test_set.distance_cat.value_counts(dropna=False,normalize=True)
#[]
#
