import numpy as np
import sklearn.svm as svm

X = np.array(
    [[-1, -1],
     [-2, -1],
     [10, 10],
     [20, 10]]
)

y = np.array([0, 0, 1, 1])

lsvm = svm.SVC(kernel='linear')

lsvm.fit(X, y)

print lsvm.coef_
print lsvm.double_coef_
print lsvm.intercept_

print

# print -lsvm.coef_[0,0]/lsvm.coef_[0,1]
# a = -lsvm.coef_[0,0]/lsvm.coef_[0,1]
# print -lsvm.intercept_[0] / lsvm.coef_[0,1]
# b = -lsvm.intercept_[0] / lsvm.coef_[0,1]

CV_X = np.array([
    [0.8, 1],
])
# print lsvm.decision_function(CV_X)

# print a
# print CV_X[0]
# print lsvm.coef_[0, 0]*CV_X[0, 0], lsvm.coef_[0, 1]*CV_X[0, 0]

d = np.sum(lsvm.coef_[0, 0]*CV_X[0, 0] + lsvm.coef_[0, 1]*CV_X[0, 1])
cv_x = np.matrix(CV_X)
coef = np.matrix(lsvm.coef_)
print 'D', d, cv_x*coef.T
print 'Coef_norm', np.linalg.norm(coef)


# print np.matrix(lsvm.coef_)*np.matrix(CV_X[0]).T
cos_theta = cv_x * coef.T / (np.linalg.norm(cv_x) * np.linalg.norm(coef))
# print cos_theta

pj_cv_x = ((cv_x*coef.T) / np.linalg.norm(coef)) * coef

ddiff = np.linalg.norm(coef) - np.linalg.norm(pj_cv_x)

print 'Projection', pj_cv_x
print 'Projection - Coef_norm', np.linalg.norm(coef) - np.linalg.norm(pj_cv_x)

new_n = pj_cv_x

print 'New_n', new_n

print 'Zero_D', cv_x*np.matrix(new_n).T
