# %%

import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import boxcox_normmax
import scipy.special as ss
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)
train["SalePrice"] = np.log1p(train["SalePrice"])
y = train['SalePrice'].reset_index(drop=True)
test_ID = test.Id
train.drop('Id', inplace=True, axis=1)
test.drop('Id', inplace=True, axis=1)
data = pd.concat((train, test)).reset_index(drop=True)

strings = ['MSSubClass', 'YrSold', 'MoSold']
for var in strings:
    data[var] = data[var].apply(str)

groups = ['Exterior1st', 'Exterior2nd', 'SaleType', 'Electrical', 'KitchenQual']
for group in groups:
    mode = data[group].mode()[0]
    data[group] = data[group].fillna(mode)

data.MSZoning = data.MSZoning.fillna('RL')

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual',
            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', "PoolQC"
    , 'Alley', 'Fence', 'MiscFeature', 'FireplaceQu', 'MasVnrType', 'Utilities']:
    data[col] = data[col].fillna('None')

for col in ('GarageArea', 'GarageCars', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2'
            , 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BsmtUnfSF', 'TotalBsmtSF', 'GarageYrBlt'):
    data[col] = data[col].fillna(0)

data.Functional = data.Functional.fillna('Typ')

data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

data.drop(['SalePrice'], axis=1, inplace=True)

data['TotalSF'] = data.TotalBsmtSF + data['1stFlrSF'] + data['2ndFlrSF']
data['TotalBath'] = data.FullBath + 0.5 * data.HalfBath + data.BsmtFullBath + 0.5 * data.BsmtHalfBath
data['TotalPorch'] = data.OpenPorchSF + data['3SsnPorch'] + data.EnclosedPorch + data.ScreenPorch + data.WoodDeckSF
data['YrBltAndRemod'] = data.YearBuilt + data.YearRemodAdd

data['HasPool'] = data.PoolArea.apply(lambda x: 1 if x > 0 else 0)
data['HasGarage'] = data.GarageArea.apply(lambda x: 1 if x > 0 else 0)
data['HasBsmt'] = data.TotalBsmtSF.apply(lambda x: 1 if x > 0 else 0)
data['HasFirePl'] = data.Fireplaces.apply(lambda x: 1 if x > 0 else 0)

drops = ['Utilities', 'Street', 'PoolQC']
data = data.drop(drops, axis=1)

cat_features = data.select_dtypes(include=['object']).columns

num_features = data.select_dtypes(exclude=['object']).columns

feat_num = data[num_features].drop(['GarageYrBlt', 'MasVnrArea'], axis=1)
feat_cat = data[cat_features]

skewness = feat_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
for feat in skewed_features:
    feat_num[feat] = ss.boxcox1p(feat_num[feat], boxcox_normmax(feat_num[feat] + 1))
    data[feat] = ss.boxcox1p(data[feat], boxcox_normmax(data[feat] + 1))

qual_dict = {"None": 0, "Po": 1, "Fa": 2, "TA": 4, "Gd": 7, "Ex": 11}
qual_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
             "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]

for cat in data.columns:
    if cat in qual_cols:
        data[cat] = data[cat].map(qual_dict).astype('int64')

fin_data = pd.get_dummies(data).reset_index(drop=True)

train = fin_data.iloc[:len(y), :]
test = fin_data.iloc[len(y):, :]

outliers = [30, 88, 462, 631, 1322]
train = train.drop(train.index[outliers])
y_train = y.drop(y.index[outliers])

y = y_train.values

from sklearn.linear_model import ElasticNet, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb

n_folds = 5

lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.00057, random_state=1))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state=7, nthread=-1)

model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf=6, min_sum_hessian_in_leaf=11)

model_xgb_deep = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                                  learning_rate=0.05, max_depth=11,
                                  min_child_weight=1.7817, n_estimators=2200,
                                  reg_alpha=0.4640, reg_lambda=0.8571,
                                  subsample=0.5213, silent=1,
                                  random_state=7, nthread=-1)

model_lgb_estimatorsH = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                                          learning_rate=0.05, n_estimators=5000,
                                          max_bin=55, bagging_fraction=0.8,
                                          bagging_freq=5, feature_fraction=0.2319,
                                          feature_fraction_seed=9, bagging_seed=9,
                                          min_data_in_leaf=3, min_sum_hessian_in_leaf=11)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    # Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)


averaged_models = AveragingModels(models=(ENet, model_xgb, model_lgb, model_xgb_deep, lasso))


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    # Do the predictions of all base models on the test data and use the averaged predictions as
    # meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_])
        return self.meta_model_.predict(meta_features)


stacked_averaged_models = StackingAveragedModels(base_models=(ENet,
                                                              model_xgb_deep,
                                                              model_xgb,
                                                              model_lgb,
                                                              model_lgb_estimatorsH),
                                                 meta_model=lasso)

stacked_averaged_models.fit(train.values, y)
model_xgb.fit(train, y)
model_lgb.fit(train, y)
lasso.fit(train, y)

xgb_pred = np.expm1(model_xgb.predict(test))
lgb_pred = np.expm1(model_lgb.predict(test.values))
lasso_pred = np.expm1(lasso.predict(test.values))
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

ensemble = stacked_pred * 0.45 + xgb_pred * 0.15 + lgb_pred * 0.3 + lasso_pred * 0.10

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('script submission.csv', index=False)
