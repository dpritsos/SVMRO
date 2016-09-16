import numpy as np
import sklearn.svm as svm

X = np.array(
    [[-1, -1, -3],
     [-2, -1, -8],
     [10, -10, -1],
     [20, 10, 50]]
)

y = np.array([0, 0, 1, 1])

lsvm = svm.LinearSVC(penalty='l2', multi_class='ovr', dual=True)
# lsvm = svm.SVC(kernel='linear', decision_function_shape='ovr')

tX = lsvm.fit_transform(X, y)

ds = lsvm.decision_function(X)
az = np.where((ds >= 0))
lz = np.where((ds <= 0))

print np.hstack((ds[lz][::-1], [0], ds[az]))

X_ds = np.hstack((ds[lz][::-1], [0], ds[az]))

s1 = 50.0
s2 = -1.0
s3 = 15.0

omega = X_ds[-1]
alpha = ds[lz][0]

print 'Omega', omega
print 'Alpha', alpha
print 'S1 omega distance', omega - s1
print 'S2 omega distance', omega - s2
print 'S2 omega distance', omega - s3
print 'S1 alpha distance', s1 - alpha
print 'S2 alpha distance', s2 - alpha
print 'S2 alpha distance', s3 - alpha

print 'Full X-Omega',  omega - X_ds
print 'Full X-Omega',  X_ds - alpha

o = omega - X_ds
a = X_ds - alpha

print np.where(((o >= 0) & (a >= 0)), 1, -1)



print np.mean(lsvm.decision_function(X))

print 'Coef', lsvm.coef_
# print 'Dual_coef', lsvm.dual_coef_
print 'Intercept', lsvm.intercept_

print

# print -lsvm.coef_[0,0]/lsvm.coef_[0,1]
# a = -lsvm.coef_[0,0]/lsvm.coef_[0,1]
# print -lsvm.intercept_[0] / lsvm.coef_[0,1]
# b = -lsvm.intercept_[0] / lsvm.coef_[0,1]

CV_X = np.array([
    [-1, -1, -3],
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
