# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.io as sio
data = sio.loadmat('spamData.mat')


# %%
def accuracy(predicted_result, actual_result):
    diff = predicted_result - actual_result
    return 1.0 - (float(np.count_nonzero(diff)) / len(diff))


# %%
def decesion_stumps(X,y,D):
    n_samples, n_features =X.shape
    F_ast = float('inf')
    for j in range(n_features):
        feature_values = X[:, j]
        # dict ={feature_values[i]:(y[i], D[i]) for i in range(n_samples)}
        sorted_features = sorted([(feature_i, y_i, D_i) for feature_i, y_i, D_i in zip(feature_values, y, D)])
        F = np.sum(D*((-y+1)/2), axis =0,  keepdims = False)

        if F<F_ast:
            F_ast = F.copy()
            theta = sorted_features[0][0] - 1
            j_ast = j
        for i in range(len(sorted_features)-1):

            feature_i, y_i ,D_i =sorted_features[i]

            F += (-D_i if y_i==-1 else D_i) 
            if F < F_ast and (feature_i != sorted_features[i+1][0]):
                F_ast =F.copy()
                theta = 0.5 *(feature_i + sorted_features[i+1][0])
                j_ast = j
    
    return j_ast, theta


# %%
def adaboost(Xtrain,ytrain,Xtest,ytest, T):
    n_samples, n_features =Xtrain.shape
    n_samples_test, n_features = Xtest.shape
    adaboost_train = []
    adaboost_test =  []
    weights = []
    err_train = []
    err_test = []

    D = 1.0/n_samples * np.ones([n_samples,1])
    for t in range(T):
        j, theta = decesion_stumps(Xtrain,ytrain,D)
        print("theta = {}".format(theta))
        print("j = {}".format(j))
        wl_train = list(map(lambda x: 1 if x > theta else -1 , Xtrain[:, j]))
        adaboost_train.append(wl_train)
    
        wl_test = list(map(lambda x: 1 if x > theta else -1, Xtest[:, j]))
        adaboost_test.append(wl_test)
        mistakes_train = [ 0 if wl_train[i] == ytrain[i] else 1 for i in range(len(ytrain))]
        dum = np.sum(mistakes_train)
        eps = np.sum(D*(np.array(mistakes_train).reshape([n_samples,1])) )
        w = 0.5*np.log(1.0/eps-1)
        weights = np.append(weights, w)
        test = -w*ytrain*np.array(wl_train).reshape(n_samples,1)
        D = D *np.exp(-w*ytrain*np.array(wl_train).reshape(n_samples,1))
        D/= np.sum(D, keepdims=False)
        if (t%10 == 0):
            dum = np.array(adaboost_train)
            dum2 = np.array(weights).reshape([len(weights),1])
            dum4 =np.array(adaboost_train)* np.array(weights).reshape([len(weights),1]) 

            dum3 = np.sum(np.array(adaboost_train)* (np.array(weights).reshape([len(weights),1])), axis =0)
            predict_train = np.sign(np.sum(np.array(adaboost_train)* (np.array(weights).reshape([len(weights),1])) , axis =0)).reshape([n_samples,1])

          
            err_train.append(1.0 - accuracy(predict_train, ytrain))
            predict_test = np.sign(np.sum(np.array(adaboost_test)* (np.array(weights).reshape([len(weights),1])) , axis =0)).reshape([ n_samples_test, 1])
            err_test.append(1.0 - accuracy(predict_test, ytest))
            print("iter: {}, training misclassification: {}".format(str(t), err_train[-1]))
            print("iter: {}, test misclassification: {}".format(str(t), err_test[-1]))
            print("theta = {}".format(theta))
            print("j = {}".format(j))
            print("w = {}".format(w))
    return err_train, err_test

# %%


# %%
Xtrain = data['Xtrain']
ytrain = data['ytrain']
Xtest = data['Xtest']
ytest = data['ytest']
#Xtrain = np.log(Xtrain +0.1)
#Xtrain = np.insert(Xtrain,[0], np.ones((mtrain,1)),axis = 1)
#Xtest = np.insert(Xtest,[0], np.ones((mtest,1)),axis = 1)
#Xtest = np.log(Xtest +0.1)
ytrain = np.int8(ytrain)*2-1
ytest = np.int8(ytest)*2-1
#ytrain = np.int8(ytrain)
#ytest = np.int8(ytest)


# %%
err_train, err_test = adaboost(Xtrain,ytrain,Xtest,ytest, T=1000)
#%matplotlib inline
fig=plt.figure()
plt.plot(err_train, label='training misclassification rate')
plt.plot(err_test, label='test misclassification rate')
plt.legend()
plt.show()

# %%



# %%


