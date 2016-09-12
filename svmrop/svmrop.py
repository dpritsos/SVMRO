
import numpy as np
import sklearn.svm as svm



class LinearSetSVM(object):

    def __init__(self, svm_type='oneclass', **kwargs):

        self.svm_type = svm_type

        if self.svm_type == 'oneclass':

            self.lsvm = svm.OneClassSVM(**kwargs)

        elif self.svm_type == 'binary':

            kwargs['kernel'] != 'linear':
                print "Warning: Only linear Kernel is supported in this SVM-extention method."
                print "Auto-config params: penalty='l2', multi_class='ovr', dual='True'"
                kwargs['penalty'] = 'l2'
                kwargs['multi_class'] = 'ovr'
                kwargs['dual'] = True

            self.lsvm = svm.LinearSVC(**kwargs)

        else:
            raise Exception("Invalid option for argument 'svm_type'")

    def optimize(self, l, a, b, X, yp, yn, yu):

        # Making an array to matrix if not already.
        X = np.matrix(X)

        # Training the Linear SVM, either one-class or binary
        if self.svm_type == 'oneclass':
            lsvm.fit(X)
        else:
            lsvm.fit(X, np.vstack(yp, yn, yu))

        # Getting preditions of the LinearSVM for all the sampels, Postive, Negative, Uknown.
        predicted_ds = lsvm.decision_function(X)

        # Sorting the predicted distances.
        predicted_ds = np.sort(predicted_ds)
        pds_inv = predicted_ds[::-1]

        # Getting the min and max distances, more or less equivalent to candidate SVs while...
        # ...the Greedy Optimization that follows.
        min_ds = np.min(predicted_ds)
        max_ds = np.max(predicted_ds)
        min_ds_i = np.argmin(predicted_ds)
        max_ds_i = np.argmax(predicted_ds)

        # Getting the Decision Hyperplane's Normal vector.
        N_vect = np.matrix(self.lsvm.coef_)

        # Getting the Normal vector of the most distaned parrale hyperplane from the...
        # ...decision hyperplane.
        max_ds_x = X[max_ds_i, :]
        proj_n_max_dsmpl = ((max_ds_x*N_vect.T) / np.linalg.norm(N_vect.T)) * N_vect
        N_far = np.cross(proj_n_max_dsmpl, max_ds_x)

        # Getting the intercept of the decision function, a.k.a the decision hyperplane.
        incpt = lsvm.intercept_

        Rs = (np.euclidiandistance(near_H) / np.sum(svm.predict(X[yp]))) +
        (np.sum(svm.predict(X[yp])) / np.euclidiandistance(near_H)) +
        pa * Va + pv * Vv

        # Starting Greedy Optimization process
        min_Risk = np.Inf
        for near_pds, far_pds in (predicted_ds, pds_inv):

            # Moving Hyperplanes.

            # Calculating Constraintes

            # Calculating Opt

    def Rs():
        return (np.euclidiandistance(near_H) / np.sum(svm.predict(X[yp]))) +
        (np.sum(svm.predict(X[yp])) / np.euclidiandistance(near_H)) +
        pa * Va + pv * Vv

    def Rem():
        p = a
        r = b
        return (1.0 / 2*((p*r)/(p+r)))







    def refine_planes(self, A, V, ppA, ppV, m, n, s):

        print svm.libsvm.decision_function()

        pass

    def predict(x):
        pass
