from sklearn.model_selection import GridSearchCV

from src.pipelines.ABCParameterSeeker import ABCParameterSeeker


class GridSearchCVSeeker(ABCParameterSeeker):

    def __init__(self, estimator, parameters, cv):
        self.seeker = GridSearchCV(estimator=estimator, param_grid=parameters, cv=cv, n_jobs=-1)

    def search(self, data, labels):
        self.seeker.fit(data, labels)
        return self.seeker.best_params_
