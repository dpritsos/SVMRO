
import numpy as np
import sklearn.svm as svm


class LinearSetSVM(object):

    def __init__(self, *args, **kwargs):

        arg_lst = ['mrgn_fw', 'mrgn_nw', 'c2_w', 'c1_w', 'll', 'svm_type']

        if args:
            for arg in args:
                self.__dict__[arg_lst.pop()] = arg

        if kwargs:
            for kwd, arg in kwargs.items():
                if kwd in arg_lst:
                    self.__dict__[arg_lst.pop()] = arg
                else:
                    msg = "Error - Argument " + kwd + " already given."
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

            self.lsvm = svm.LinearSVC(**kwargs)

        else:
            raise Exception("Invalid option for argument 'svm_type'")

    def optimize(self, X, yp_i, yn_i, yu_i):

        # Training the Linear SVM, either one-class or binary
        if self.svm_type == 'oneclass':
            print 'FIT'
            self.lsvm.fit(X[yp_i, :])

        else:
            y = np.zeros(yp_i.shape[0] + yn_i.shape[0], dtype=np.int)
            y[0:yp_i.shape[0]] = 1
            y[yp_i.shape[0]:yp_i.shape[0]+yn_i.shape[0]] = -1

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
                p4y_pos = self.predictions(X[yp_i], new_near, new_far)
                p4y_neg = self.predictions(X[yn_i], new_near, new_far)

                hl_pos = np.array([np.max([0, 1.0 - f]) for f in p4y_pos])
                hl_neg = np.array([np.max([0, 1.0 - f]) for f in p4y_neg])

                cstr_1 = yp_i.shape[0]*self.c1_w <= np.sum(hl_pos)
                cstr_2 = yn_i.shape[0]*self.c2_w >= np.sum(hl_neg)

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
                    Re = 1.0 / 2.0*((p*r)/(p+r))

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
            near_H = self.pds[self.near_H_i]*(0.5 - self.mrgn_nw) +\
                self.pds[self.near_H_i - 1]*(self.mrgn_nw - 0.5)
        else:
            near_H = np.min(self.pds) - self.mrgn_nw*ds_posz

        if (self.far_H_i+1) < (yp_i.shape[0] + yn_i.shape[0]):
            far_H = self.pds[self.far_H_i]*(0.5 - self.mrgn_fw) +\
                self.pds[self.far_H_i + 1]*(self.mrgn_fw - 0.5)
        else:
            far_H = np.max(self.pds) - self.mrgn_fw*ds_posz

        return near_H, far_H

    def predict(self, X, near_H, far_H):

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

        self.gnr_classes = dict()

        super(SVMRO, self).__init__(*args, **kwargs)

        # Required params for SVM
        # Linear penalty='l2', multi_class='ovr', dual=True
        # OneClass nu

    def fit(self, trn_idxs, corpus_mtrx, cls_gnr_tgs, params):

        print "TRAINING"

        inds_per_gnr = dict()
        inds = list()
        last_gnr_tag = 1

        for gnr_tag in np.unique(cls_gnr_tgs[trn_idxs]):
            inds_per_gnr[self.genres_lst[gnr_tag - 1]] = trn_idxs[
                np.where(cls_gnr_tgs[trn_idxs] == gnr_tag)[0]
            ]

        for g, inds in inds_per_gnr.items():

            """
            # Create the OC-SVM Model for this genre
            if params['svm_type'] == 'oneclass':
                self.lopsvm = svmrop.LinearSetSVM(
                    svm_type='oneclass',
                    ll=params['ll'],
                    c1_w=params['c1_w'], c2_w=params['c2_w'],
                    mrgn_nw=params['mrgn_nw'], mrgn_fw=params['mrgn_fw'],
                    nu=params['nu']
                )
            else:
                self.lopsvm = svmrop.LinearSetSVM(
                    svm_type='binary',
                    ll=params['ll'],
                    c1_w=params['c1_w'], c2_w=params['c2_w'],
                    mrgn_nw=params['mrgn_nw'], mrgn_fw=params['mrgn_fw'],

                )
            """
            # print "Fit Model for ", g
            # print "Corpus_Mtrx", corpus_mtrx[inds_per_gnr[g], :].shape
            # Convert TF vectors to Binary
            # crp_arr_bin = np.where(corpus_mtrx[inds_per_gnr[g], :].toarray() > 0, 1, 0)

            # ### Fitting Linear 1-vs-set SVM Model to Data of this genre. ###

            # Getting the positive samples for this split
            yp_i = inds

            # Getting some negative samples for this split.
            # print inds_per_gnr.items()
            # print [inds for grn, inds in inds_per_gnr.items() if grn != g]
            yn_i = np.hstack(
                [
                    np.random.permutation(inds)[0:int(np.floor(inds.shape[0]/2.0))]
                    for grn, inds in inds_per_gnr.items() if grn != g
                ]
            )

            # Not in use yet!
            yu_i = 0

            # Training the SVM 1-vs-set
            near_H, far_H = self.optimize(
                corpus_mtrx[:, 0:params['features_size']], yp_i, yn_i, yu_i
            )

            # Refining the Hyperplanes based on the user expirience on the data set.
            near_H, far_H = self.refine_planes(
                corpus_mtrx[:, 0:params['features_size']], yp_i, yn_i
            )

            # Keeping the Decision function, i.e. the decision hyperplanes for the evaluation.
            self.gnr_classes[g] = [near_H, far_H]

        return self.gnr_classes

    def predict(self, *args):

        print 'PREDICTING'

        # Get Input arguments in given sequence
        crv_idxs = args[0]
        corpus_mtrx = args[1]
        params = args[2]

        # Get the part of matrices or arrays required for the model prediction phase
        crossval_X = corpus_mtrx[crv_idxs, 0:params['features_size']]

        # Get the part of matrices required for the model prediction phase.
        # ###crossval_Y =  cls_gnr_tgs [crv_idxs, :]

        # Initialize Predicted-Classes-Arrays List
        predicted_Y_per_gnr = list()
        predicted_d_near_per_gnr = list()
        predicted_d_far_per_gnr = list()
        gnr_cls_idx = list()

        for g in self.gnr_classes.keys():

            # Converting TF vectors to Binary
            # cv_arr_bin = np.where(crossval_X.toarray() > 0, 1, 0)

            # Getting the predictions for each Vector for this genre
            predicted_Y, predicted_D_near, predicted_D_far = self.lopsvm.predict(
                crossval_X, self.gnr_classes[g][0], self.gnr_classes[g][1]
            )

            # Keeping the prediction per genre
            predicted_Y_per_gnr.append(predicted_Y)
            predicted_d_near_per_gnr.append(predicted_D_near)
            predicted_d_far_per_gnr.append(predicted_D_far)
            gnr_cls_idx.append(self.genres_lst.index(g) + 1)

        # Converting it to Array before returning
        predicted_Y_per_gnr = np.vstack(predicted_Y_per_gnr)
        predicted_d_near_per_gnr = np.vstack(predicted_d_near_per_gnr)
        predicted_d_far_per_gnr = np.vstack(predicted_d_far_per_gnr)
        gnr_cls_idx = np.hstack(gnr_cls_idx)

        return predicted_Y_per_gnr, predicted_d_near_per_gnr, predicted_d_far_per_gnr, gnr_cls_idx


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
