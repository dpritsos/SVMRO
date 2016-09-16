
import numpy as np
import sklearn.svm as svm



class LinearSetSVM(object):

    def __init__(self, svm_type='oneclass', l, a, b, mrgn_nw, mrgn_fw, **kwargs):

        self.l = l
        self.a = a
        self.b = b
        self.mrgn_nw = mrgn_nw
        self.mrgn_fw = mrgn_fw

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
        pds = lsvm.decision_function(X)

        # Sorting the predicted distances.
        pds_idxs = np.argsort(pds)
        inv_pds_idxs = pds_idxs[::-1]

        # Getting index in the pds vector of the min and max distances, more or less...
        # equivalent to candidate SVs for the Greedy Optimization that follows. Then setting...
        # ...them as the initill Close and Far Hyperplane intial ofset position form the...
        # ...SVM hyperplane/decision function.
        self.self.near_H_i = np.argmin(pds)
        self.self.far_H_i = np.argmax(pds)

        # Getting the Decision Hyperplane's Normal vector.
        # self.N_vect = np.matrix(self.lsvm.coef_)

        # Getting the Normal vector of the most distaned parrale hyperplane from the...
        # ...decision hyperplane.
        # max_ds_x = X[max_ds_i, :]
        # proj_n_max_dsmpl = ((max_ds_x*self.near_H.T) / np.linalg.norm(self.near_H.T)) *...
        # self.near_H
        # self.far_H = np.cross(proj_n_max_dsmpl, max_ds_x)

        # Getting the intercept of the decision function, a.k.a the decision hyperplane.
        # incpt = lsvm.intercept_
        # I probably won't use this.

        # Starting Greedy Optimization process
        min_Risk = np.Inf
        for pds_i in np.random.shuffle(pds_idxs[1::]):
            for inv_pds_i in random.shuffle(inv_pds_idxs[1::]):

                # Moving the Hyperplanes.
                new_near = pds[pds_i]
                new_far = pds[inv_pds_i]

                # ## Calculating Constraintes ##
                f4y_pos = self.dfunc(X[yp_i], new_near, new_far)
                f4y_neg = self.dfunc(X[yn_i],  new_near, new_far)

                hl_pos = np.array([np.max([0, 1.0 - f]) for f in f4y_pos])
                hl_neg = np.array([np.max([0, 1.0 - f]) for f in f4y_neg])

                cstr_1 = yp_i.shape[0]*self.a <= np.sum(hl_pos)
                cstr_2 = yn_i.shape[0]*self.b >= np.sum(hl_neg)

                if cstr_1 and cstr_2:

                    # ## Calculating Opt ##

                    # Getting prediciton fo the current position of the Hyperplanes.
                    pre_y = self.predictions(Χ, new_near, new_far)

                    # Forming the expected Y vector.
                    exp_y = np.zeros_like(pre_y)
                    exp_y[:] = -9
                    exp_y[yp_i] = 1
                    exp_y[yn_i] = -1

                    # Calculating True Postive, False Postive and False Negative for this...
                    # ...Hyperplane setup.
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

                    # Caculating Empirical Risk which is been selected to be the inverced...
                    # ...F-measure,eg F1.
                    Re = 1.0 / 2.0*((p*r)/(p+r))

                    # ## Caclulating the Open Space Risk Rs ##
                    dz_from_f = np.abs(pds - new_near)
                    dz_from_f = dz_from_f[np.where((dz_from_f > 0))]
                    margin_N = np.min(dz_from_f)

                    dz_from_n = np.abs(pds - new_near)
                    dz_from_n = dz_from_f[np.where((dz_from_f > 0))]
                    margin_N = np.min(dz_from_n)

                    ds_posz = self.dfunc(X[yp_i], new_near, new_far)

                    Rs = ((margin_F - margin_N) / ds_posz) + (ds_posz / (margin_F - margin_N))
                    # Plus Margin spaces which is a bit vague...
                    Rs = self.mrgn_nw*margin_N + self.mrgn_fw*margin_F

                    if min_Risk > (Rs + Re):
                        min_Risk = Rs + Re
                        self.near_H_i = pds_i
                        self.far_H_i = inv_pds_i

        return pds[self.near_H_i], pds[self.far_H_i]

    def refine_planes(self, X, m, n, s):

        print svm.libsvm.decision_function()

        pass

    def predictions(self, Χ, near_H, far_H):

        dsz_N, dsz_F = self.dfunc(X, near_H, far_H)

        return np.where(((dsz_N >= 0) & (dsz_F >= 0)), 1, -1)

    def dfunc(self, X, near_H, far_H):

        dsz_X = self.lsvm.decision_function(X)

        dsz_N = far_H - dsz_X
        dsz_F = dsz_X - near_H

        return dsz_N, dsz_F

    def _Rs():
        pass

    def _Re():
        pass
