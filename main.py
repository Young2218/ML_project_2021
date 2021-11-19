from autoML import AutoML

from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, plot_confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import pickle, joblib


def getContainList(list, key) -> list:
    result = []
    for word in list:
        if key in word:
            result.append(word)
    
    return result

def preprocessing_11():
    # read dataset
    dataset = pd.read_csv('Dataset/mpii_human_pose.csv')

    category_df = dataset
    category_df['new'] = np.nan

    cate_list = dataset['Activity'].astype('category').values.categories

    # bicycling    
    category_df.loc[category_df['Category'] == 'bicycling', 'new'] = 'bicycling' #516
    category_df.loc[category_df['Activity'] == 'bicycling, stationary', 'new'] = 'bicycling' # 102
    category_df.loc[category_df['Activity'] == 'upper body exercise, stationary bicycle - Airdyne (arms only) 40 rpm, moderate', 'new'] = 'bicycling' # 32
    print('bicycling: ' + str(len(category_df.loc[category_df['new'] == 'bicycling', :]))) # 650

    # # yoga
    category_df.loc[category_df['Activity'] == 'yoga, Power', 'new'] = 'yoga' 
    category_df.loc[category_df['Activity'] == 'yoga, Nadisodhana', 'new'] = 'yoga'
    category_df.loc[category_df['Activity'] == 'stretching', 'new'] = 'yoga'
    print('yoga: ' + str(len(category_df.loc[category_df['new'] == 'yoga', :]))) # 340

    # manmom
    category_df.loc[category_df['Activity'] == 'video exercise workouts, TV conditioning programs', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'resistance training', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'circuit training', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'aerobic, step', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'pilates, general', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'calisthenics', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'home exercise, general', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'slide board exercise, general', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'stretching', 'new'] = 'manmom'
    category_df.loc[category_df['Activity'] == 'rope skipping, general', 'new'] = 'manmom'
    print('manmom: ' + str(len(category_df.loc[category_df['new'] == 'manmom', :]))) # 1024

    # rowing
    category_df.loc[category_df['Activity'] == 'rowing, stationary', 'new'] = 'rowing' 
    print('rowing: ' + str(len(category_df.loc[category_df['new'] == 'rowing', :]))) # 150


    # skiing
    ski_list = getContainList(cate_list, 'skiing')
    category_df.loc[((category_df['Activity'].isin(ski_list)) & 
                    (category_df['Category'] == 'winter activities')), 'new'] = 'skiing' 
    print('ski: ' + str(len(category_df.loc[category_df['new'] == 'skiing', :]))) # 355

    # running
    category_df.loc[category_df['Category'] == 'running', 'new'] = 'running' 
    print('running: ' + str(len(category_df.loc[category_df['new'] == 'running', :]))) # 291

    # skateboarding
    category_df.loc[category_df['Activity'] == 'skateboarding', 'new'] = 'skateboarding' 
    print('skateboarding: ' + str(len(category_df.loc[category_df['new'] == 'skateboarding', :]))) # 184
    
    # baseball
    category_df.loc[category_df['Activity'] == 'softball, general', 'new'] = 'baseball'  
    print('baseball: ' + str(len(category_df.loc[category_df['new'] == 'baseball', :]))) # 173

    # soccer
    category_df.loc[category_df['Activity'] == 'soccer', 'new'] = 'soccer'  
    print('soccer: ' + str(len(category_df.loc[category_df['new'] == 'soccer', :]))) # 137

    # golf
    category_df.loc[category_df['Activity'] == 'golf', 'new'] = 'golf'
    print('golf: ' + str(len(category_df.loc[category_df['new'] == 'golf', :]))) # 138

    # basketball
    category_df.loc[category_df['Activity'] == 'basketball', 'new'] = 'basketball'
    category_df.loc[category_df['Activity'] == 'basketball, game (Taylor Code 490)', 'new'] = 'basketball'
    print('basketball: ' + str(len(category_df.loc[category_df['new'] == 'basketball', :]))) # 170


    category_df.dropna(axis=0, inplace=True)
    X = category_df.drop(columns=['ID', 'NAME', 'Scale', 'Activity', 'Category', 'new'])
    y = category_df['new']

    scaledX = pd.DataFrame(StandardScaler().fit_transform(X.transpose()))
    # encode_y = LabelEncoder().fit_transform(y)


    sm = SMOTE()
    x_resample, y_resample = sm.fit_resample(scaledX.transpose(), y)

    return x_resample, y_resample

def preprocessing_4():
    # read dataset
    dataset = pd.read_csv('Dataset/mpii_human_pose.csv')

    category_df = dataset
    category_df['new'] = np.nan

    cate_list = dataset['Activity'].astype('category').values.categories

     # yoga
    category_df.loc[category_df['Activity'] == 'yoga, Power', 'new'] = 'yoga' 
    category_df.loc[category_df['Activity'] == 'yoga, Nadisodhana', 'new'] = 'yoga'
    category_df.loc[category_df['Activity'] == 'stretching', 'new'] = 'yoga'
    print('yoga: ' + str(len(category_df.loc[category_df['new'] == 'yoga', :]))) # 340

    # rowing
    category_df.loc[category_df['Activity'] == 'rowing, stationary', 'new'] = 'rowing' 
    print('rowing: ' + str(len(category_df.loc[category_df['new'] == 'rowing', :]))) # 150

    # running
    category_df.loc[category_df['Category'] == 'running', 'new'] = 'running' 
    print('running: ' + str(len(category_df.loc[category_df['new'] == 'running', :]))) # 291

    # golf
    category_df.loc[category_df['Activity'] == 'golf', 'new'] = 'golf'
    print('golf: ' + str(len(category_df.loc[category_df['new'] == 'golf', :]))) # 138


    category_df.dropna(axis=0, inplace=True)
    X = category_df.drop(columns=['ID', 'NAME', 'Scale', 'Activity', 'Category', 'new'])
    y = category_df['new']

    scaledX = pd.DataFrame(StandardScaler().fit_transform(X.transpose()))
    # encode_y = LabelEncoder().fit_transform(y)


    sm = SMOTE()
    x_resample, y_resample = sm.fit_resample(scaledX.transpose(), y)

    return x_resample, y_resample

def clustering2Graphic_11(x,y):

    rgbs = plt.cm.get_cmap('Set3')
    colors = []
    for c in rgbs.colors:
        colors.append(rgb_to_hex(c[0], c[1], c[2]))

    # kmeans with n_culster = 3 /////////////////////////////////
    py = KMeans(n_clusters=3).fit_predict(x)
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(1)
    for c in range(3):
        plt.subplot(1,3,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        vc.sort_index(inplace=True)
        plt.pie(vc, labels=vc.index, colors=colors)
    
    # kmeans with n_culster = 5 /////////////////////////////////
    py = KMeans(n_clusters=5).fit_predict(x)
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(2)
    for c in range(5):
        plt.subplot(2,3,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        plt.pie(vc, labels=vc.index,colors=colors)
    
    # gm with n_culster = 5 /////////////////////////////////
    py = GaussianMixture(n_components=5, covariance_type='spherical').fit_predict(x)
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(3)
    for c in range(5):
        plt.subplot(2,3,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        vc.sort_index(inplace=True)
        plt.pie(vc, labels=vc.index, colors=colors)


    # agglomerative with n_culster = 3 /////////////////////////////////
    py = AgglomerativeClustering(n_clusters=3, linkage='single').fit_predict(x)
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(4)
    for c in range(3):
        plt.subplot(1,3,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        vc.sort_index(inplace=True)
        plt.pie(vc, labels=vc.index, colors=colors)

    # mean shift /////////////////////////////////
    py = MeanShift(bandwidth=3.662).fit_predict(x)
    cl_len = len(np.unique(py))
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(5)
    for c in range(cl_len):
        plt.subplot(cl_len/5+1,5,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        vc.sort_index(inplace=True)
        plt.pie(vc, labels=vc.index, colors=colors)




    plt.show()

def clustering2Graphic_4(x,y):

    rgbs = plt.cm.get_cmap('Set3')
    colors = []
    for c in rgbs.colors:
        colors.append(rgb_to_hex(c[0], c[1], c[2]))

    # kmeans with n_culster = 3 /////////////////////////////////
    py = KMeans(n_clusters=3).fit_predict(x)
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(1)
    for c in range(3):
        plt.subplot(1,3,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        vc.sort_index(inplace=True)
        plt.pie(vc, labels=vc.index, colors=colors)
    
    # kmeans with n_culster = 8 /////////////////////////////////
    py = KMeans(n_clusters=8).fit_predict(x)
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(2)
    for c in range(5):
        plt.subplot(3,3,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        plt.pie(vc, labels=vc.index,colors=colors)
    
    # gm with n_culster = 6 /////////////////////////////////
    py = GaussianMixture(n_components=6, covariance_type='spherical').fit_predict(x)
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(3)
    for c in range(5):
        plt.subplot(2,3,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        vc.sort_index(inplace=True)
        plt.pie(vc, labels=vc.index, colors=colors)


    # agglomerative with n_culster = 4 /////////////////////////////////
    py = AgglomerativeClustering(n_clusters=4, linkage='single').fit_predict(x)
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(4)
    for c in range(3):
        plt.subplot(1,3,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        vc.sort_index(inplace=True)
        plt.pie(vc, labels=vc.index, colors=colors)

    # mean shift /////////////////////////////////
    py = MeanShift(bandwidth=4.097).fit_predict(x)
    cl_len = len(np.unique(py))
    pre_df = pd.DataFrame({'cluster':py,'target':y})

    plt.figure(5)
    for c in range(cl_len):
        plt.subplot(cl_len/5+1,5,c+1)
        plt.title('Cluster {}'.format(c+1))
        vc = pre_df[pre_df['cluster'] == c]['target'].value_counts()
        vc.sort_index(inplace=True)
        plt.pie(vc, labels=vc.index, colors=colors)




    plt.show()

def rgb_to_hex(r, g, b):
    r, g, b = int(255*r), int(255*g), int(255*b)
    return '#' + hex(r)[2:].zfill(2) + hex(g)[2:].zfill(2) + hex(b)[2:].zfill(2)

def getPurity(y, y_pred) -> float:
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


if __name__ == '__main__':
    
    # preprocssing
    x, y = preprocessing_4()
    print('data length: {}'.format(len(x)))

    atml = AutoML(x,y)

    result = ''
    result += 'DecisionTree: ' + str(atml.autoDecisionTree(1, 30)) + '\n'
    result += 'Logistic Regression: ' + str(atml.autoLogisticReg()) + '\n'
    result += 'SVM: ' + str(atml.autoSVM()) + '\n'
    result += 'KMeans: ' + str(atml.autoKmeans(1, 2, 30, 0.85)) + '\n'
    result += 'Gaussian Mixture: ' + str(atml.autoGM(1, 2, 30)) + '\n'
    result += 'Aggolmerative Clustering: ' + str(atml.autoAggolmerative(1, 2, 30)) + '\n'
    result += 'Mean Shift: ' + str(atml.autoMeanshift()) + '\n'
    print(result)
        
    
    result = ''
    for model in [DecisionTreeClassifier(), SVC(), LogisticRegression()] :
        model.fit(atml.X_train,atml.y_train)
        score = model.score(atml.X_test,atml.y_test)
        result += '{}: {}\n'.format(model, score)
    print(result)
    
    result = ''
    for model in [KMeans(), MeanShift(), GaussianMixture(n_components=4), AgglomerativeClustering()]:
        pred = model.fit_predict(atml.X)
        score = silhouette_score(atml.X, pred)
        result += '{} silhouette score: {}\n'.format(model, score)
        score = getPurity(atml.y, pred)
        result += '{} purity score: {}\n'.format(model, score)
    print(result)

