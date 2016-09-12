import numpy as np
import sklearn.svm as svm

X = np.array(
    [[-1, -1, -3],
     [-2, -1, -8],
     [10, 10, 10],
     [20, 10, 50]]
)

y = np.array([0, 0, 1, 1])

lsvm = svm.LinearSVC(penalty='l2', multi_class='ovr', dual=True)
# lsvm = svm.SVC(kernel='linear', decision_function_shape='ovr')

lsvm.fit(X, y)

print 'Coef', lsvm.coef_
# print 'Dual_coef', lsvm.dual_coef_
print 'Intercept', lsvm.intercept_

print

# print -lsvm.coef_[0,0]/lsvm.coef_[0,1]
# a = -lsvm.coef_[0,0]/lsvm.coef_[0,1]
# print -lsvm.intercept_[0] / lsvm.coef_[0,1]
# b = -lsvm.intercept_[0] / lsvm.coef_[0,1]

CV_X = np.array([
    [-5, -1, -6],
])
# print lsvm.decision_function(CV_X)

# print a
# print CV_X[0]
# print lsvm.coef_[0, 0]*CV_X[0, 0], lsvm.coef_[0, 1]*CV_X[0, 0]

cv_x = np.matrix(CV_X)
coef = np.matrix(lsvm.coef_)
d = np.sum(coef[0, 0]*cv_x[0, 0] + coef[0, 1]*cv_x[0, 1] + coef[0, 2]*cv_x[0, 2])+lsvm.intercept_
print 'Decision_Function', lsvm.decision_function(cv_x)
print 'D', d, (cv_x*coef.T)+lsvm.intercept_
print 'Coef_norm', np.linalg.norm(coef)


# print np.matrix(lsvm.coef_)*np.matrix(CV_X[0]).T
cos_theta = cv_x * coef.T / (np.linalg.norm(cv_x) * np.linalg.norm(coef))
# print cos_theta

pj_cv_x = ((cv_x*coef.T) / np.linalg.norm(coef)) * coef
print
print 'Projection', pj_cv_x
print 'Coef', coef

print pj_cv_x, cv_x
new_n = np.cross(pj_cv_x[0], cv_x[0])

print 'New_n', new_n

print 'Zero_D#1', cv_x*np.matrix(new_n).T
print 'Zero_D#2', pj_cv_x*np.matrix(new_n).T
