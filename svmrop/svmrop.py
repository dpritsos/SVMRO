
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
                print "Auto-config kernel to Linear kernel."
                kwargs['kernel'] = 'linear'

                # kwargs['decision_function_shape'] = ’ovr’ <-- Consider it.

            self.lsvm = svm.SVC(**kwargs)

        else:
            raise Exception("Invalid option for argument 'svm_type'")

    def optimize(self, l, a, b, X, yp, yn, yu):

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

        # Getting the Decition Hyperplane a.k.a Decision Function.
        near_H = svm.libsvm.decision_function()
        far_H = near_H + max_ds  # It is not working yet, it moves the near_H Hyperplane to max_ds.

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
