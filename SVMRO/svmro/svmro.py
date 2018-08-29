
import numpy as np
import sklearn.svm as svm


class LinearSetSVM(object):

    def __init__(self, *args, **kwargs):

        arg_lst = ['mrgn_fw', 'mrgn_nw', 'c2_w', 'c1_w', 'll', 'svm_type']

        if args:

            if len(args) > 6:
                cargs = [arg for arg in args[0:6]]
                args = args[5::]

            for arg in cargs:
                self.__dict__[arg_lst.pop()] = arg

        if kwargs:
            for kwd, arg in kwargs.items():
                if kwd in arg_lst:

                    self.__dict__[kwd] = arg

                    arg_lst.remove(kwd)

                    del kwargs[kwd]

                elif kwd in self.__dict__.keys():
                    msg = "Error - Argument " + kwd + \
                        " already given with value: " + str(self.__dict__[kwd])
                    raise Exception(msg)

        if arg_lst:
            msg = "Argument(s) " + ", ".join(arg_lst) + 'missing.'
            raise Exception(msg)

        if self.svm_type == 'oneclass':

            print "Warning: Only linear Kernel is supported in this SVM-extention method."
            kwargs['kernel'] = 'linear'
            self.lsvm = svm.OneClassSVM(*args, **kwargs)

        elif self.svm_type == 'binary':

            print "Warning: Only linear Kernel is supported in this SVM-extention method."
            print "Auto-config params: penalty='l2', multi_class='ovr', dual='True'"
            kwargs['penalty'] = 'l2'
            kwargs['multi_class'] = 'ovr'
            kwargs['dual'] = True

            self.lsvm = svm.LinearSVC(*args, **kwargs)

        else:
            raise Exception("Invalid option for argument 'svm_type'")

    def optimize(self, X, yp_i, yn_i, yu_i):

        print 'FIT SKLEARN SVM'
        # Training the Linear SVM, either one-class or binary
        if self.svm_type == 'oneclass':
            self.lsvm.fit(X[yp_i, :])

        else:
            y = np.zeros(yp_i.shape[0] + yn_i.shape[0], dtype=np.int)
            y[0:yp_i.shape[0]] = 1
            y[yp_i.shape[0]:yp_i.shape[0] + yn_i.shape[0]] = -1

            self.lsvm.fit(X[np.hstack((yp_i, yn_i)), :], y)

        # Getting preditions of the LinearSVM for all the sampels, Postive, Negative, Uknown.
        self.pds = self.lsvm.decision_function(X[np.hstack((yp_i, yn_i)), :])
        if self.svm_type == 'oneclass':
            self.pds = np.transpose(self.pds)[0]

        # Sorting the predicted distances.
        pds_idxs = np.argsort(self.pds)
        inv_pds_idxs = pds_idxs[::-1]

        # Getting index in the pds vector of the min and max distances, more or less...
        # equivalent to candidate SVs for the Greedy Optimization that follows. Then setting...
        # ...them as the initill Close and Far Hyperplane intial ofset position form the...
        # ...SVM hyperplane/decision function.
        self.near_H_i = np.argmin(self.pds)
        self.far_H_i = np.argmax(self.pds)

        print 'Optimization'
        # Starting Greedy Optimization process
        min_Risk = np.Inf
        c = 0
        while c < 1:

            c += 1
            pdsidxs = np.random.permutation(pds_idxs[1::])
            invpdsidxs = np.random.permutation(inv_pds_idxs[1::])

            for pds_i, inv_pds_i in zip(pdsidxs, invpdsidxs):

                # Moving the Hyperplanes.
                new_near = self.pds[pds_i]
                new_far = self.pds[inv_pds_i]

                # ## Calculating Constraintes ##
                p4y_pos = self.predict__(X[yp_i], new_near, new_far)[0]
                p4y_neg = self.predict__(X[yn_i], new_near, new_far)[0]

                hl_pos = np.array([np.max([0, 1.0 - f]) for f in p4y_pos])
                hl_neg = np.array([np.max([0, 1.0 - f]) for f in p4y_neg])

                cstr_1 = yp_i.shape[0] * self.c1_w <= np.sum(hl_pos)
                cstr_2 = yn_i.shape[0] * self.c2_w >= np.sum(hl_neg)

                if cstr_1 and cstr_2:

                    # ## Calculating Opt ##

                    # Getting prediciton fo the current position of the Hyperplanes.
                    pre_y = np.hstack((p4y_pos, p4y_neg))
                    # pre_y = self.predictions(X[np.hstack((yp_i, yn_i)), :], new_near, new_far)

                    # Forming the expected Y vector.
                    exp_y = np.zeros_like(pre_y)
                    exp_y[:] = -9
                    exp_y[0:yp_i.shape[0]] = 1
                    exp_y[yp_i.shape[0]:yn_i.shape[0]] = -1

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
                    Re = 1.0 / 2.0 * ((p * r) / (p + r))

                    # ## Caclulating the Open Space Risk Rs ##
                    dz_from_n = np.abs(self.pds - new_near)
                    dz_from_n = dz_from_n[np.where((dz_from_n > 0))]
                    margin_N = np.min(dz_from_n)

                    dz_from_f = np.abs(self.pds - new_near)
                    dz_from_f = dz_from_f[np.where((dz_from_f > 0))]
                    margin_F = np.min(dz_from_f)

                    # BE CAREFULL self.dfunc(X[yp_i], new_near, new_far) OR
                    ds_posz = np.sum(self.lsvm.decision_function(X[yp_i]))
                    # print ds_posz

                    Rs = ((margin_F - margin_N) / ds_posz) + (ds_posz / (margin_F - margin_N))
                    # Plus Margin spaces which is a bit vague...
                    Rs = self.mrgn_nw*margin_N + self.mrgn_fw*margin_F

                    if min_Risk > (Rs + self.ll * Re):
                        min_Risk = Rs + self.ll * Re
                        self.near_H_i = pds_i
                        self.far_H_i = inv_pds_i
                    print 'Culc Done'
        print 'Optimization Done'
        return self.pds[self.near_H_i], self.pds[self.far_H_i]

    def refine_planes(self, X, yp_i, yn_i):

        ds_posz = np.sum(self.lsvm.decision_function(X[yp_i]))

        if self.near_H_i > 0:
            near_H = self.pds[self.near_H_i] * (0.5 - self.mrgn_nw) +\
                self.pds[self.near_H_i - 1] * (self.mrgn_nw - 0.5)
        else:
            near_H = np.min(self.pds) - self.mrgn_nw*ds_posz

        if (self.far_H_i + 1) < (yp_i.shape[0] + yn_i.shape[0]):
            far_H = self.pds[self.far_H_i] * (0.5 - self.mrgn_fw) +\
                self.pds[self.far_H_i + 1] * (self.mrgn_fw - 0.5)
        else:
            far_H = np.max(self.pds) - self.mrgn_fw * ds_posz

        return near_H, far_H

    def predict__(self, X, near_H, far_H):

        dsz_N, dsz_F = self.dfunc(X, near_H, far_H)

        return np.where(((dsz_N >= 0) & (dsz_F >= 0)), 1, -1), dsz_N, dsz_F

    def dfunc(self, X, near_H, far_H):

        dsz_X = self.lsvm.decision_function(X)
        if self.svm_type == 'oneclass':
            dsz_X = np.transpose(dsz_X)[0]

        dsz_N = far_H - dsz_X
        dsz_F = dsz_X - near_H

        return dsz_N, dsz_F


class SVMRO(LinearSetSVM):

    def __init__(self, *args, **kwargs):

        arg_lst = ['mrgn_fw', 'mrgn_nw', 'c2_w', 'c1_w', 'll', 'svm_type']

        self.ci2gtag = dict()
        self.gnr_classes = dict()

        super(SVMRO, self).__init__(*args, **kwargs)

        # Required params for SVM
        # Linear penalty='l2', multi_class='ovr', dual=True
        # OneClass nu

    def fit(self, corpus_mtrx, cls_tgs):

        print "TRAINING"

        for i, gnr_tag in enumerate(np.unique(cls_tgs)):

            self.ci2gtag[i] = gnr_tag

            # Getting the positive samples for this split
            yp_i = np.where((cls_tgs == gnr_tag))[0]

            # Getting some negative samples for this split.
            yn_i = np.random.permutation(np.where((cls_tgs != gnr_tag))[0])
            yn_i = yn_i[0:int(np.floor(yn_i.size / 2.0))]

            # Not in use yet!
            yu_i = 0

            # Training the SVM 1-vs-set
            near_H, far_H = self.optimize(corpus_mtrx, yp_i, yn_i, yu_i)

            # Refining the Hyperplanes based on the user expirience on the data set.
            near_H, far_H = self.refine_planes(corpus_mtrx, yp_i, yn_i)

            # Keeping the Decision function, i.e. the decision hyperplanes for the evaluation.
            self.gnr_classes[i] = [near_H, far_H]

        return self.gnr_classes

    def predict(self, corpus_mtrx):

        print 'PREDICTING'

        # Get the part of matrices required for the model prediction phase.
        # ###crossval_Y =  cls_gnr_tgs [crv_idxs, :]

        # Initialize Predicted-Classes-Arrays List
        pre_Y_bin_per_gnr = list()
        pre_Dn_per_gnr = list()
        pre_Df_per_gnr = list()

        for i in np.arange(len(self.ci2gtag)):

            # Getting the predictions for each Vector for this genre
            pre_Y_bin, preD_n, preD_f = self.predict__(
                corpus_mtrx, self.gnr_classes[i][0], self.gnr_classes[i][1]
            )

            # Keeping the prediction per genre
            pre_Y_bin_per_gnr.append(pre_Y_bin)
            pre_Dn_per_gnr.append(preD_n)
            pre_Df_per_gnr.append(preD_f)

        # Converting it to Array before returning
        pYbin_per_gnr = np.vstack(pre_Y_bin_per_gnr)
        pDn_per_gnr = np.vstack(pre_Dn_per_gnr)
        pDf_per_gnr = np.vstack(pre_Df_per_gnr)

        # MAX
        D_scores = pDn_per_gnr + pDf_per_gnr
        maxD_i = np.argmax(D_scores, axis=0)
        maxScore_Cls = np.array([self.ci2gtag[i] for i in maxD_i])

        predicted_Y = np.zeros(pYbin_per_gnr.shape[1])

        posPred_j = np.unique(np.where((pYbin_per_gnr > 0))[1])

        predicted_Y[posPred_j] = maxScore_Cls[posPred_j]

        return predicted_Y, pDf_per_gnr, pDn_per_gnr, np.max(D_scores, axis=0)


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
    print 'Predicted Y', llsvm.predict(X, near_H, far_H)

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
