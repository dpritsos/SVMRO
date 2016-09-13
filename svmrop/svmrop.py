
import numpy as np
import sklearn.svm as svm



class LinearSetSVM(object):

    def __init__(self, svm_type='oneclass', l, a, b, ppN, ppF, **kwargs):

        self.l = l
        self.a = a
        self.b = b
        ppN = ppN
        ppF = ppF

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

    def optimize(self, X, yp_i, yn_i, yu_i):

        # Making an array to matrix if not already.
        X = np.matrix(X)

        # Training the Linear SVM, either one-class or binary
        if self.svm_type == 'oneclass':
            lsvm.fit(X)
        else:
            lsvm.fit(X, np.vstack(yp, yn))

        # Getting preditions of the LinearSVM for all the sampels, Postive, Negative, Uknown.
        predicted_ds = lsvm.decision_function(X)

        # Sorting the predicted distances.
        pds_idxs = np.argsort(predicted_ds)
        inv_pds_idxs = pds_idxs[::-1]

        # Getting the min and max distances, more or less equivalent to candidate SVs while...
        # ...the Greedy Optimization that follows.
        min_ds = np.min(predicted_ds)
        max_ds = np.max(predicted_ds)
        min_ds_i = np.argmin(predicted_ds)
        max_ds_i = np.argmax(predicted_ds)

        # Getting the Decision Hyperplane's Normal vector.
        self.N_vect = np.matrix(self.lsvm.coef_)

        # Getting the Normal vector of the most distaned parrale hyperplane from the...
        # ...decision hyperplane.
        max_ds_x = X[max_ds_i, :]
        proj_n_max_dsmpl = ((max_ds_x*N_vect.T) / np.linalg.norm(N_vect.T)) * N_vect
        self.N_far = np.cross(proj_n_max_dsmpl, max_ds_x)

        # Getting the intercept of the decision function, a.k.a the decision hyperplane.
        incpt = lsvm.intercept_

        # Starting Greedy Optimization process
        min_Risk = np.Inf
        for pds_i, inv_pds_i in (pds_idxs, inv_pds_idxs):
            pass


            # Moving Hyperplanes.

            # ## Calculating Constraintes ##
            f4y_pos = self.dfunc(X[yp_i])
            f4y_neg = self.dfunc(X[yn_i])

            

            cstr_1 = yp_i.shape[0]*self.a <= np.sum(np.max([0, 1-z]))
            cstr_2 = yn_i.shape[0]*self.b >= np.sum(np.max([0, 1-z]))


            # ## Calculating Opt ##
            # Getting prediciton fo the carrent position of the Hyperplanes.
            pre_y = self.predictions(Χ)

            # Forming the expected Y vector.
            exp_y = np.zeros_like(pre_y)
            exp_y[:] = -9
            exp_y[yp_i] = 1
            exp_y[yn_i] = -1

            # Calculating True Postive, False Postive and False Negative for this Hyperplane setup.
            tp = np.sum(
                np.where(
                    ((pre_y == exp_y) & (exp_y == 1) & (exp_y != -9)),
                    1.0, 0.0
                )
            )
            fp = np.sum(
                np.where(
                    ((pre_y != exp_y) & (exp_y == -1) & (exp_y != -9)),
                    1.0, 0.0
                )
            )
            fn = np.sum(
                np.where(
                    ((pre_y != exp_y) & (exp_y == 1) & (exp_y != -9)),
                    1.0, 0.0
                )
            )

            # Calculating the Precision Recall fir this hyperplane setup.
            p = tp / (tp + fp)
            r = tp / (tp + fn)

            # Caculating Empirical Risk which is been selected to be the inverced F-measure,eg F1.
            Re = 1.0 / 2.0*((p*r)/(p+r))

        return

    def refine_planes(self, A, V, ppA, ppV, m, n, s):

        print svm.libsvm.decision_function()

        pass

    def predictions(Χ):

        preds = np.zeros((X.shape[0]), dtype=np.int)

        for i, x in enumerate(X):

            dsz_N, dsz_F = self.dfunc(X)

            # NOTE: The theorethical condition is "A <= f(x) and f(x) <= V".
            if dsz_N >= 0 and dsz_F <= 0:
                preds[i] += 1
            else:
                preds[i] -= 1

        return preds

    def dfunc(X):

        dsz_N = np.zeros((X.shape[0]), dtype=np.float)
        dsz_F = np.zeros((X.shape[0]), dtype=np.float)

        for i, x in enumerate(X):

            dsz_N[i] = x*self.N_vect
            dsz_F[i] = x*self.N_far

        return dsz_N, dsz_F


    def _precision(self, pre_y, yp, yn):





    def _recall(self, pre_y, yp, yn):

    def _Rs():
        return (np.euclidiandistance(near_H) / np.sum(svm.predict(X[yp]))) +
        (np.sum(svm.predict(X[yp])) / np.euclidiandistance(near_H)) +
        pa * Va + pv * Vv

    def _Re():
        p = a
        r = b
        return (1.0 / 2*((p*r)/(p+r)))
