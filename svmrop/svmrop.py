
import numpy as np
import sklearn.svm as svm


class LinearSetSVM(object):

    def __init__(self, svm_type, l, c1_w, c2_w, mrgn_nw, mrgn_fw, **kwargs):

        self.l = l
        self.c1_w = c1_w
        self.c2_w = c2_w
        self.mrgn_nw = mrgn_nw
        self.mrgn_fw = mrgn_fw

        self.svm_type = svm_type

        if self.svm_type == 'oneclass':

            print "Warning: Only linear Kernel is supported in this SVM-extention method."
            kwargs['kernel'] = 'linear'
            self.lsvm = svm.OneClassSVM(**kwargs)

        elif self.svm_type == 'binary':

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
        #X = np.matrix(X)

        # Training the Linear SVM, either one-class or binary
        if self.svm_type == 'oneclass':

            self.lsvm.fit(X[yp_i, :])

        else:
            y = np.zeros(yp_i.shape[0] + yn_i.shape[0], dtype=np.int)
            y[yp_i] = 1
            y[yn_i] = -1

            self.lsvm.fit(X[np.hstack((yp_i, yn_i)), :], y)

        # Getting preditions of the LinearSVM for all the sampels, Postive, Negative, Uknown.
        pds = self.lsvm.decision_function(X)

        # Sorting the predicted distances.
        pds_idxs = np.argsort(pds)
        inv_pds_idxs = pds_idxs[::-1]

        # Getting index in the pds vector of the min and max distances, more or less...
        # equivalent to candidate SVs for the Greedy Optimization that follows. Then setting...
        # ...them as the initill Close and Far Hyperplane intial ofset position form the...
        # ...SVM hyperplane/decision function.
        self.near_H_i = np.argmin(pds)
        self.far_H_i = np.argmax(pds)

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
        for pds_i in pds_idxs[1::]:  # np.random.shuflle maybe ?
            for inv_pds_i in inv_pds_idxs[1::]:

                # Moving the Hyperplanes.
                new_near = pds[pds_i]
                new_far = pds[inv_pds_i]

                # ## Calculating Constraintes ##
                p4y_pos = self.predictions(X[yp_i], new_near, new_far)
                p4y_neg = self.predictions(X[yn_i], new_near, new_far)

                hl_pos = np.array([np.max([0, 1.0 - f]) for f in p4y_pos])
                hl_neg = np.array([np.max([0, 1.0 - f]) for f in p4y_neg])

                cstr_1 = yp_i.shape[0]*self.c1_w <= np.sum(hl_pos)
                cstr_2 = yn_i.shape[0]*self.c2_w >= np.sum(hl_neg)

                if cstr_1 and cstr_2:

                    # ## Calculating Opt ##

                    # Getting prediciton fo the current position of the Hyperplanes.
                    pre_y = self.predictions(X, new_near, new_far)

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

                    # BE CAREFULL self.dfunc(X[yp_i], new_near, new_far) OR
                    ds_posz = np.sum(self.lsvm.decision_function(X[yp_i]))
                    print ds_posz

                    Rs = ((margin_F - margin_N) / ds_posz) + (ds_posz / (margin_F - margin_N))
                    # Plus Margin spaces which is a bit vague...
                    Rs = self.mrgn_nw*margin_N + self.mrgn_fw*margin_F

                    if min_Risk > (Rs + self.l*Re):
                        min_Risk = Rs + self.l*Re
                        self.near_H_i = pds_i
                        self.far_H_i = inv_pds_i

        return pds[self.near_H_i], pds[self.far_H_i]

    def refine_planes(self, yp_i, yn_i):

        #
        pds = self.lsvm.decision_function(X)

        # BE CAREFULL self.dfunc(X[yp_i], new_near, new_far) OR
        ds_posz = np.sum(self.lsvm.decision_function(X[yp_i]))

        if self.near_H_i > 0:
            near_H = pds[self.near_H_i]*(0.5 - self.mrgn_nw) +\
                pds[self.near_H_i - 1]*(self.mrgn_nw - 0.5)
        else:
            near_H = np.min(pds) - self.mrgn_nw*ds_posz

        if self.far_H_i < (yp_i.shape[0] + yn_i.shape[0]):
            far_H = pds[self.far_H_i]*(0.5 - self.mrgn_fw) +\
                pds[self.far_H_i + 1]*(self.mrgn_fw - 0.5)
        else:
            far_H = np.max(pds) - self.mrgn_fw*ds_posz

        return near_H, far_H

    def predictions(self, X, near_H, far_H):

        dsz_N, dsz_F = self.dfunc(X, near_H, far_H)

        return np.where(((dsz_N >= 0) & (dsz_F >= 0)), 1, -1)

    def dfunc(self, X, near_H, far_H):

        dsz_X = self.lsvm.decision_function(X)

        dsz_N = far_H - dsz_X
        dsz_F = dsz_X - near_H

        return dsz_N, dsz_F


if __name__ == '__main__':

    X = np.array(
        [
            [-2, -1],
            [-1, -3],
            [-2, -3],
            [1, 1],
            [1, 2],
            [3, 1],
            [3, 5],
            [100, 10],
            [120, 90]
        ]
    )

    yn_i = np.array([0, 1, 2])
    yp_i = np.array([3, 4, 5, 6])
    yu_i = np.array([6, 7])

    y = np.zeros(yn_i.shape[0] + yp_i.shape[0] + yu_i.shape[0])
    y[yp_i] = 1.0
    y[yn_i] = -1.0

    llsvm = LinearSetSVM(
        svm_type='oneclass', l=0.8, c1_w=0.5, c2_w=0.5, mrgn_nw=0.3, mrgn_fw=0.3,
        # penalty='l2', multi_class='ovr', dual=True
        nu=0.08
    )

    near_H, far_H = llsvm.optimize(X, yp_i, yn_i, yu_i)

    near_H, far_H = llsvm.refine_planes(yp_i, yn_i)

    print 'Data set X:', X
    print 'Expected Y:', y
    print 'Predicted Y', llsvm.predictions(X, near_H, far_H)
