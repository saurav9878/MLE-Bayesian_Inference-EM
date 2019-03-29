import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, multivariate_normal, beta, binom
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

# Data Visualization

df = pd.read_excel('irisdata.xls', header=None)

Encoder = LabelEncoder()
Y = Encoder.fit_transform(df.iloc[:,4])

plt.scatter(df.iloc[:50,3], df.iloc[:50,1],c='red')
plt.scatter(df.iloc[50:100,3],df.iloc[50:100,1],c='blue')
plt.scatter(df.iloc[100:150,3], df.iloc[100:150,1], c='green')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Class 0', 'Class 1', 'Class 2'])


# Decorrelation

zero_mean_X = stats.zscore(df.iloc[:,:2])
covarience = np.cov(zero_mean_X.T)
eigenValues, eigenVectors = np.linalg.eig(covarience)
decorrelated = np.matmul(zero_mean_X, eigenVectors)
print(np.cov(decorrelated.T))

# The covarience of decorrelated data is a DIAGONAL matrix as obtained above

plt.scatter(decorrelated[:50,0], decorrelated[:50,1],c='red')
plt.scatter(decorrelated[50:100,0],decorrelated[50:100,1],c='blue')
plt.scatter(decorrelated[100:150,0], decorrelated[100:150,1], c='green')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Class 0', 'Class 1', 'Class 2'])

# Whitening
# We've added a small value (here $10^{-5}$) to avoid the division by 0

whitened_data = decorrelated/np.sqrt(eigenValues+1e-5)
print(np.cov(whitened_data.T))

# The covarience of Whitened data is an IDENTITY matrix as obtained above

plt.scatter(whitened_data[:50,0], whitened_data[:50,1],c='red')
plt.scatter(whitened_data[50:100,0],whitened_data[50:100,1],c='blue')
plt.scatter(whitened_data[100:150,0], whitened_data[100:150,1], c='green')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Class 0', 'Class 1', 'Class 2'])

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:4], Y, test_size=0.3, random_state=1)


#Classes are equiprobable, so prior probabilities are same for all

Prior_prob = [1/3, 1/3, 1/3]

X_class_0 = X_train.loc[y_train == 0]
X_class_1 = X_train.loc[y_train == 1]
X_class_2 = X_train.loc[y_train == 2]

mean_class_0 = X_class_0.mean()
sigma_class_0 = X_class_0.std()

mean_class_1 = X_class_1.mean()
sigma_class_1 = X_class_1.std()

mean_class_2 = X_class_2.mean()
sigma_class_2 = X_class_2.std()

def classify_using_naive_bayes(X):
    likelihood = np.zeros((len(X[0]),3))
    likelihood[:,0] = norm(mean_class_0[0],sigma_class_0[0]).pdf(X[0])*norm(mean_class_0[1], sigma_class_0[1]).pdf(X[1])*norm(mean_class_0[2], sigma_class_0[2]).pdf(X[2])
    likelihood[:,1] = norm(mean_class_1[0],sigma_class_1[0]).pdf(X[0])*norm(mean_class_1[1], sigma_class_1[1]).pdf(X[1])*norm(mean_class_1[2], sigma_class_1[2]).pdf(X[2])
    likelihood[:,2] = norm(mean_class_2[0],sigma_class_2[0]).pdf(X[0])*norm(mean_class_2[1], sigma_class_2[1]).pdf(X[1])*norm(mean_class_2[2], sigma_class_2[2]).pdf(X[2])
    Posterior = Prior_prob*likelihood
    pred = np.argmax(Posterior,axis=1)
    return pred


y_pred = classify_using_naive_bayes(X_train)
training_accuracy = np.mean(y_pred==y_train)
print(training_accuracy*100)

Y_pred = classify_using_naive_bayes(X_test)
test_accuracy = np.mean(Y_pred==y_test)
print(test_accuracy*100)


# Maximum Likelihood Estimation

likelihood = np.zeros((len(X_train[0]),3))
likelihood[:,0] = multivariate_normal.pdf(X_train,mean = mean_class_0, cov = X_class_0.cov())
likelihood[:,1] = multivariate_normal.pdf(X_train,mean = mean_class_1, cov = X_class_1.cov())
likelihood[:,2] = multivariate_normal.pdf(X_train,mean = mean_class_2, cov = X_class_2.cov())

def classify_using_bayes(X_test):
    likelihood = np.zeros((len(X_test[0]),3))
    likelihood[:,0] = multivariate_normal.pdf(X_test,mean = mean_class_0, cov = X_class_0.cov())
    likelihood[:,1] = multivariate_normal.pdf(X_test,mean = mean_class_1, cov = X_class_1.cov())
    likelihood[:,2] = multivariate_normal.pdf(X_test,mean = mean_class_2, cov = X_class_2.cov())
    Posterior = Prior_prob*likelihood
    y_pred = np.argmax(Posterior, axis=1)
    return y_pred

Prior_prob = [1/3, 1/3, 1/3]
y_pred = classify_using_bayes(X_test)
test_accuracy = np.mean(y_pred==y_test)
print(test_accuracy*100)

Prior_prob = [0.3, 0.5, 0.2]
y_pred = classify_using_bayes(X_test)
test_accuracy = np.mean(y_pred==y_test)
print(test_accuracy*100)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:4], Y, test_size=0.5, random_state=1)

X_class_0 = X_train.loc[y_train == 0]
X_class_1 = X_train.loc[y_train == 1]
X_class_2 = X_train.loc[y_train == 2]

mean_class_0 = X_class_0.mean()
mean_class_1 = X_class_1.mean()
mean_class_2 = X_class_2.mean()

Prior_prob = [1/3, 1/3,1/3]
y_pred = classify_using_bayes(X_test)
test_accuracy = np.mean(y_pred==y_test)
print(test_accuracy*100)

X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,0:4], Y, test_size=0.7, random_state=1)

X_class_0 = X_train.loc[y_train == 0]
X_class_1 = X_train.loc[y_train == 1]
X_class_2 = X_train.loc[y_train == 2]

mean_class_0 = X_class_0.mean()
mean_class_1 = X_class_1.mean()
mean_class_2 = X_class_2.mean()

Prior_prob = [1/3, 1/3,1/3]
y_pred = classify_using_bayes(X_test)
test_accuracy = np.mean(y_pred==y_test)
print(test_accuracy*100)


# The accuracy of ML estimate is 100% until model is training and test set size is 50%. When test size increases more than training set, accuracy starts to decrease from 100%

# Probability Distributions

a=0.1
b=0.1
X = np.linspace(0+1e-5,1-1e-5,10000) #beta dist is only defined for x = [0,1]
rv = beta(a,b)
plt.plot(X,rv.pdf(X))
print("Mean is ", np.mean(rv.pdf(X)))
print("Variance is ", np.var(rv.pdf(X)))

a=1
b=1
X = np.linspace(0,1,10000)
rv = beta(a,b)
plt.plot(X,rv.pdf(X))
print("Mean is ", np.mean(rv.pdf(X)))
print("Variance is ", np.var(rv.pdf(X)))

a=2
b=3
X = np.linspace(0,1,10000)
rv = beta(a,b)
plt.plot(X,rv.pdf(X))
print("Mean is ", np.mean(rv.pdf(X)))
print("Variance is ", np.var(rv.pdf(X)))

a=8
b=4
X = np.linspace(0,1,10000)
rv = beta(a,b)
plt.plot(X,rv.pdf(X))
print("Mean is ", np.mean(rv.pdf(X)))
print("Variance is ", np.var(rv.pdf(X)))

n = 50
p = 0.5
X = np.arange(0,10000)
y = binom.pmf(X,n,p)
plt.plot(X,y)
print("Mean is ", np.mean(y))
print("Variance is ", np.var(y))

n=100
p=0.5
X = np.arange(10000)
y = binom.pmf(X,n,p)
plt.plot(X,y)
print("Mean is ", np.mean(y))
print("Variance is ", np.var(y))

n=50
p=0.7
X = np.arange(10000)
y = binom.pmf(X,n,p)
plt.plot(X,y)
print("Mean is ", np.mean(y))
print("Variance is ", np.var(y))

# Conjugate Priors and Bayesian Inference

X = np.linspace(0,1,1000)
prior = beta.pdf(X,2,2)
k=4
n=5
likelihood = binom.pmf(k,n,X)
posterior = prior*likelihood
plt.plot(posterior)

k=1
n=1
likelihood = binom.pmf(k,n,X)
posterior = prior*likelihood
plt.plot(posterior)

points = np.random.normal(0.8,0.1,100)
prior = norm.pdf(points,0,1)
likelihood = norm.pdf(points,points.mean(),0.1)
posterior = prior*likelihood
order = np.argsort(points)
X = np.array(points)[order]
Y = np.array(posterior)[order]
plt.plot(X,Y)

points = np.random.normal(0.8,0.1,1000)
prior = norm.pdf(points,0,1)
likelihood = norm.pdf(points,points.mean(),0.1)
posterior = prior*likelihood
order = np.argsort(points)
X = np.array(points)[order]
Y = np.array(posterior)[order]
plt.plot(X,Y)


# When sample points are changed from 100 to 1000, the variance has decreased

print("Posterior Mean is ",posterior.mean())
print("Posterior Varience is ", posterior.var())

# GMM and Expectation Maximization Algorithm

X =[]
for i in range(500):
    if i%4==0 or i%4==1:
        value = np.random.normal(1,0.1,1)
        X = np.append(X,value)
    elif i%4==2:
        value = np.random.normal(3,0.1,1)
        X = np.append(X,value)
    else:
        value = np.random.normal(2,0.2,1)
        X = np.append(X,value)

gmm = GaussianMixture(n_components=3)
gmm = gmm.fit(X.reshape(-1,1))
means = gmm.means_
variances = gmm.covariances_
weights = A.weights_

print(means)
print(variances)
print(weights)
