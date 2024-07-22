from typing import Tuple, List
import numpy as np
import pandas as pd
import logging
import sklearn
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LassoCV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from ..util.logger_config import setup_logger
LOG_LEVEL = logging.INFO
logger = setup_logger(__name__, level = logging.INFO)

LASSO_CV_REGRESSOR = LassoCV(cv = 5, random_state = 0, max_iter = 500000)

def try_model(X, y, polys = 2, regressor = LASSO_CV_REGRESSOR) -> Tuple[sklearn.pipeline.Pipeline, float, float, float]:
    '''
    Returns
    -------
      Model, R^2 on training data, R^2 on test data
    '''
    X_train, X_test  = X
    y_train, y_test = y
    X_train = X_train.reset_index().drop('index', axis = 1)
    y_train = np.array(y_train, dtype = float)
    model = make_pipeline(SimpleImputer(missing_values = np.nan, strategy = 'mean'), PolynomialFeatures(polys), MaxAbsScaler(), regressor)
    logger.info("Fitting model...")
    start = time.time()
    model.fit(X_train, y_train)
    time_to_finish = time.time() - start
    logger.info("Finished fitting model in %s seconds", time_to_finish)
    print(f"R^2 score on training data {model.score(X_train, y_train)}")
    print(f"R^2 score on test data {model.score(X_test, y_test)}")
    print()
    return model, model.score(X_train, y_train), model.score(X_test, y_test), time_to_finish

def split_and_try_model(df : pd.DataFrame, 
                        y_var : str,  
                        x_vars : List[str], 
                        polys : int = 2, 
                        regressor = LASSO_CV_REGRESSOR,
                        random_state: int = 0
                        ) -> Tuple[sklearn.pipeline.Pipeline, float, float, float]:
    '''
    Returns
    -------
      Model, R^2 on training data, R^2 on test data, time to finish
    '''
    reg_train, reg_test = train_test_split(df, test_size = 0.2, random_state = random_state)
    base_x : Tuple[pd.DataFrame, pd.DataFrame] = (reg_train[x_vars].copy(), reg_test[x_vars].copy())
    y_pts : Tuple[pd.DataFrame, pd.DataFrame] = (reg_train[y_var].copy(), reg_test[y_var].copy())
    return try_model(base_x, y_pts, polys = polys, regressor = regressor)