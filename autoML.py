from typing import Any
import numpy as np
from numpy.random import gamma
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import datasets

from imblearn.over_sampling import SMOTE


class AutoML:
    def __init__(self, X:pd.DataFrame, y, test_size = 0.2) -> None:
        self.X = X
        self.y = y

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=1)
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def autoDecisionTree(self, min_d: int, max_d:int):
        """
        find best depth of decision tree using grid search

        Param
        --------
        min_d: int
            minimum depth of decision tree

        max_d: int
            maximum depth of decision tree

        Return
        --------
        dict:
                return best depth for grid search
            None means can not find best d in given range(min_d, max_d)
            {'best depth ' = __, 'best score ' = __}
        """
        X_train = self.X
        y_train = self.y
        param_grid = [{'max_depth': np.arange(min_d, max_d)}]
        dt_gscv = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=10)
        dt_gscv.fit(X_train, y_train)
        
        result = {'Best depth': dt_gscv.best_params_, 'Best score':dt_gscv.best_score_}

        return result

    def autoLogisticReg(self, learning_rate = 0.5, stack = 5, max_iter = 100):
        """
            find best parameters of logistic regression 

            Param
            --------
            learning_rate: float, default = 0.5
                learning rate of C in logistic regression
            
            stack: int, default = 5
                number of repeat to exit to local minima
            
            max_iter: int, default = 100
                maximun number of iteration that do not fall in infinite loop

            Return
            --------
            dict:
                return best score and best c
                {'best score ' = __, 'best penalty ' = __, 'best C ' = __ }
        """
        global_best_score = 0
        global_best_param = [0]
        for penalty in ['none', 'l2']:
            penalty_best_score = 0
            penalty_best_param = [0]
            for c_learning in [learning_rate, -learning_rate]:

                c_stack = 0
                c = 1.0
                c_best_score = 0
                c_best_param = None
                c_iter = 0

                while c_stack < stack and c > 0 and c_iter < max_iter:
                    model = LogisticRegression(penalty=penalty, C=c)
                    model.fit(self.X_train, self.y_train)
                    score = model.score(self.X_test, self.y_test)

                    if score > c_best_score:
                        c_best_score = score
                        c_best_param = (penalty, c)
                        c_stack = 0
                    else:
                        c_stack += 1

                    c += c_learning * c
                    c_iter += 1
                
                # update
                if c_best_score > penalty_best_score:
                    penalty_best_score = c_best_score
                    penalty_best_param[0] = c_best_param
            
            if penalty_best_score > global_best_score:
                global_best_score = penalty_best_score
                global_best_param[0] = penalty_best_param[0]

        penalty, c = global_best_param[0]
        return {'best score':global_best_score, 'best penalty':penalty, 'best C':c}

    def autoSVM(self, c_lr = 0.5, gamma_lr = 0.5, stack = 5, max_iter = 100):
        """
            find best parameters of SVM

            Param
            -------
            c_lr: float, default = 0.5
                learning rate of c in SVC model
            
            gamma_lr: float, default = 0.5
                learning rate of gamma in SVC model
            
            stack: int, default = 5
                number of repeat to exit to local minima
            
            max_iter: int, default = 100
                maximun number of iteration that do not fall in infinite loop

            Return
            --------
            dict:
                return best score and best c
                {'best score ' = __, 'best kernel ' = __, 'best C ' = __ ,'best gamma ' = __}
        """
        c_max_iter = max_iter
        gamma_max_iter = max_iter

        global_best_score = 0
        global_best_param = [0]

        for kernel in ['rbf', 'poly', 'sigmoid']:
            kernel_best_score = 0
            kernel_best_param = [0]

            for gamma_learing in [gamma_lr, -gamma_lr]:
                gamma = 0.01
                gamma_stack = 0
                gamma_best_score = 0
                gamma_best_param = [0]
                gamma_iter = 0
                while gamma_stack < stack and gamma_max_iter > gamma_iter:
                    
                    for c_learning in [c_lr, -c_lr]:
                        c_stack = 0
                        c = 0.1
                        c_best_score = 0
                        c_best_param = None
                        c_iter = 0

                        while c_stack < stack and c > 0 and c_iter < c_max_iter:
                            model = SVC(C=c, kernel=kernel, gamma=gamma)
                            model.fit(self.X_train, self.y_train)
                            score = model.score(self.X_test, self.y_test)

                            if score > c_best_score:
                                c_best_score = score
                                c_best_param = (c, gamma, kernel)
                                c_stack = 0
                            else:
                                c_stack += 1

                            c += c_learning * c
                            c_iter += 1
                        
                        # update
                        if c_best_score > gamma_best_score:
                            gamma_best_score = c_best_score
                            gamma_best_param[0] = c_best_param
                    
                    if gamma_best_score > kernel_best_score:
                        kernel_best_score = gamma_best_score
                        kernel_best_param[0] = gamma_best_param[0]
                        gamma_stack = 0
                    else:
                        gamma_stack += 1
                    
                    gamma += gamma_learing*gamma
                    gamma_iter +=1
        
            if kernel_best_score > global_best_score:
                global_best_score = kernel_best_score
                global_best_param[0] = kernel_best_param[0]
        
        c, gamma, kernel = global_best_param[0]
        result = {'best score': global_best_score, 'best C': c, 'best gamma': gamma, 'best kernel': kernel}
        return result


    def autoKmeans(self, learning_k: int,min_k: int, max_k: int, elbow_threshold: float) -> dict:
        """
            find best k of kmeans using elbow method, silhouette score and purity score

            Param
            --------
            learing_k: int
                number of add to k, k += learning_k

            min_k: int 
                minimum k of kmeans

            max_k: int
                maxiumn k of kmeans

            elbow_threshold: float 
                threshold to find elbow point

            Return
            --------
            dict: 
                return best k for elbow and silhouette
                None means can not find best k in given range(min_k, max_k)
                {'best k for elbow' = __ , 
                'best k for silhouette': __ , 'best silhouette score': __,
                'best k for purity': __ , 'best purity score': __}
        """
        k = min_k

        # elbow method
        elbow_best_k = False
        inertias = []

        # silhouette values
        silhouette_best_k = False
        silhouette_scores = []
        sil_max_score = 0
        sil_stack = 0   

        # purity values
        purity_max_score = 0
        purity_stack = 0
        purity_best_k = 0

        while ((not elbow_best_k) or sil_stack < 4 or purity_stack < 4) and k < max_k:
            model = KMeans(n_clusters=k)
            model.fit(self.X)
            pred = model.fit_predict(self.X)

            # elbow
            if not elbow_best_k:
                inertias.append(model.inertia_)

                if len(inertias) >= 3:
                    now = model.inertia_
                    prev = inertias[k-3]
                    pprev = inertias[k-4]

                    if (now - prev) < elbow_threshold*(prev - pprev):
                        elbow_best_k = k - 1

            # silhouette
            si_score = silhouette_score(self.X, pred)
            silhouette_scores.append(si_score)
            if max(silhouette_scores) > sil_max_score:
                sil_max_score = max(silhouette_scores)
                silhouette_best_k = k
                sil_stack = 0
            else:
                sil_stack += 1


            # purity
            score = self.getPurity(self.y, pred, k)
            if score > purity_max_score:
                purity_max_score = score
                purity_best_k = k
                purity_stack = 0
            else:
                purity_stack += 1    


            k += learning_k

        # packing result to return
        if elbow_best_k == False:
            elbow_best_k = None
        if silhouette_best_k == False:
            silhouette_best_k = None

        result = {}
        result['best k for elbow'] = elbow_best_k
        result['best k for silhouette'] = silhouette_best_k
        result['best silhouette score'] = max(silhouette_scores)
        result['best k for purity'] = purity_best_k
        result['best purity score'] = purity_max_score

        return result

    def autoGM(self, learning_k: int, min_k: int, max_k: int) -> dict:
        """
        find best k of gaussian mixture's parameters using silhouette score and purity

        Param
        --------
        learning_k: int
                number of add to k, k += learning_k

        min_k: int 
            minimum k of gaussian mixture n componets

        max_k: int
            maxiumn k of gaussian mixture n componets

        Return
        --------
        dict:
            return best n components and best covarinace type and score
            that calculate by silhouette score and purity
        """
        k = min_k
        silhouette_dict = {}
        purity_dict = {}

        for covar_type in ['full', 'tied', 'diag', 'spherical']:
            # silhouette values
            silhouette_best_k = False
            silhouette_scores = []
            sil_max_score = 0
            sil_stack = 0   

            # purity values
            purity_max_score = 0
            purity_stack = 0
            purity_best_k = 0

            while (sil_stack < 4 or purity_stack < 4) and k < max_k:
                model = GaussianMixture(n_components=k, covariance_type=covar_type)
                model.fit(self.X)
                pred = model.fit_predict(self.X)
                
                # silhouette
                si_score = silhouette_score(self.X, pred)
                silhouette_scores.append(si_score)
                if max(silhouette_scores) > sil_max_score:
                    sil_max_score = max(silhouette_scores)
                    silhouette_best_k = k
                    sil_stack = 0
                else:
                    sil_stack += 1
                
               # purity
                score = self.getPurity(self.y, pred, k)
                if score > purity_max_score:
                    purity_max_score = score
                    purity_best_k = k
                    purity_stack = 0
                else:
                    purity_stack += 1 

                
                k += learning_k

            
            # outer while
            silhouette_dict[sil_max_score] = (silhouette_best_k, covar_type)
            purity_dict[purity_max_score] = (purity_best_k, covar_type)
            

        # packing result to return
        result = {}
        max_score = max(silhouette_dict.keys())
        k, covar_type = silhouette_dict[max_score]
        result['best k for silhouette'] = k
        result['best covariance type for silhouette'] = covar_type
        result['best silhouette score'] = max_score

        max_score = max(purity_dict.keys())
        k, covar_type = purity_dict[max_score]
        result['best k for purity'] = k
        result['best covariance type for purity'] = covar_type
        result['best purity score'] = max_score

        return result

    def autoAggolmerative(self, learning_k: int, min_k: int, max_k: int) -> dict:
        """
            find best k of aggolmerative clustering parameters using silhouette score and purity

            Param
            --------
            learning_k: int
                    number of add to k, k += learning_k

            min_k: int 
                minimum k of aggolmerative clustering n componets

            max_k: int
                maxiumn k of aggolmerative clustering n componets

            Return
            --------
            dict:
                return best n clusters and best linkage and score
                that calculate by silhouette score and purity
        """
        k = min_k
        silhouette_dict = {}
        purity_dict = {}

        for linkage in ['ward', 'complete', 'average', 'single']:
            # silhouette values
            silhouette_best_k = False
            silhouette_scores = []
            sil_max_score = 0
            sil_stack = 0   

            # purity values
            purity_max_score = 0
            purity_stack = 0
            purity_best_k = 0

            while (sil_stack < 4 or purity_stack < 4) and k < max_k:
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                model.fit(self.X)
                pred = model.fit_predict(self.X)
                
                # silhouette
                si_score = silhouette_score(self.X, pred)
                silhouette_scores.append(si_score)
                if max(silhouette_scores) > sil_max_score:
                    sil_max_score = max(silhouette_scores)
                    silhouette_best_k = k
                    sil_stack = 0
                else:
                    sil_stack += 1
                
               # purity
                score = self.getPurity(self.y, pred, k)
                if score > purity_max_score:
                    purity_max_score = score
                    purity_best_k = k
                    purity_stack = 0
                else:
                    purity_stack += 1 

                k += learning_k

            # outer while
            silhouette_dict[sil_max_score] = (silhouette_best_k, linkage)
            purity_dict[purity_max_score] = (purity_best_k, linkage)
            

        # packing result to return
        result = {}
        max_score = max(silhouette_dict.keys())
        sk, slinkage = silhouette_dict[max_score]
        result['best k for silhouette'] = k
        result['best linkage for silhouette'] = linkage
        result['best silhouette score'] = max_score

        max_score = max(purity_dict.keys())
        k, linkage = purity_dict[max_score]
        result['best k for purity'] = k
        result['best linkage for purity'] = linkage
        result['best purity score'] = max_score
        return result

    def autoMeanshift(self, learning_rate = 0.5, stack = 5, max_iter = 100):
        """
            find best bandwidth for Mean Shift
             Param
            --------
            learning_rate: float
                learning rate to control bandwidth

            stack: int 
                number of chance that exit local minima

            max_iter: int
                maxiumn iteration number to prevent infinite loop

            Return
            --------
            dict:
                return best n clusters and best linkage and score
                that calculate by silhouette score and purity
        """
        global_best_sil_score = 0
        global_best_purity_score = 0

        for lr in [learning_rate, -learning_rate]:
            bandwidth = 1.0
            sil_stack = 0
            sil_max_score = 0
            purity_stack = 0
            purity_max_score = 0
            c_iter = 0
            silhouette_scores = []

            while (purity_stack < stack or sil_stack < stack) and bandwidth  > 0 and c_iter < max_iter:
                model = MeanShift(bandwidth=bandwidth)
                pred = model.fit_predict(self.X)
                if len(pd.Series(pred).unique()) == 1:
                    sil_stack += 1
                    purity_stack += 1
                    continue

                # silhouette
                si_score = silhouette_score(self.X, pred)
                silhouette_scores.append(si_score)
                if max(silhouette_scores) > sil_max_score:
                    sil_max_score = max(silhouette_scores)
                    silhouette_best_bandwidth = bandwidth
                    sil_stack = 0
                else:
                    sil_stack += 1
                
               # purity
                score = self.getPurity(self.y, pred, 2)
                if score > purity_max_score:
                    purity_max_score = score
                    purity_best_bandwidth = bandwidth
                    purity_stack = 0
                else:
                    purity_stack += 1

                bandwidth += lr * bandwidth
                c_iter += 1
            
            # update
            if sil_max_score > global_best_sil_score:
                global_best_sil_score = sil_max_score
                global_best_sil_bandwidth = silhouette_best_bandwidth

            if purity_max_score > global_best_purity_score:
                global_best_purity_score = purity_max_score
                global_best_purity_bandwidth = purity_best_bandwidth
        
        return {'best silhouette score': global_best_sil_score, 'best silhouette bandwidth':global_best_sil_bandwidth, 
                'best purity score': global_best_purity_score, 'best purity bandwidth': global_best_purity_bandwidth}
        
    @staticmethod
    def getPurity(y, y_pred, k:int) -> float:
        """
         calculate cluster's purity given y and y_pred

         Param
         ---------
         y: original target class
         y_pred: cluster information

         Return
         ----------
        """
        df = pd.DataFrame(np.array([y,y_pred]).transpose(), columns=['real', 'pred'])
        k = len(pd.Series(y_pred).unique())
        sum = 0
        for i in range(k):
            vc = df[df['pred'] == i]['real'].value_counts()
            sum += max(vc)
        
        purity = sum/len(y)

        return purity
