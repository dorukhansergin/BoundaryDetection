import os
import multiprocessing as mp
from statistics import median
import time

import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

class Experiment:
    def __init__(self, images_dir, column_annotations_dir, cv=5, n_jobs=1):

        self.images_dir = images_dir
        self.column_annotations_dir = column_annotations_dir
        self.filename_list = list(map(lambda filename: filename.split('.')[0],
                                 os.listdir(os.path.join(column_annotations_dir))))
        self.cv = cv
        if n_jobs > mp.cpu_count():
            raise ValueError("Specified number of cores ({0}) is more than cpu cores ({1})"
                             .format(n_jobs, mp.cpu_count()))
        elif n_jobs == -1:
            n_jobs = mp.cpu_count()

        self.n_jobs = n_jobs
        self.features_dict = None
        self.labels_dict = None

    def set_pipeline(self, pipe_list):
        self.pipe = Pipeline(pipe_list)




class WCExperiment(Experiment):
    def __init__(self, feat_extractor, margin, column_annotations_dir, images_dir, cv=2, n_jobs=1):
        super(WCExperiment, self).__init__(images_dir, column_annotations_dir, cv, n_jobs)
        self.feat_extractor = feat_extractor(margin)
        self.feat_extractor.set_directories(column_annotations_dir, images_dir)

    def run_nested_cv(self, pipe, hyperparam_grid):
        pipe = pipe

        kfold = KFold(n_splits=self.cv)
        scores = []
        for idx_trn, idx_tst in kfold.split(self.filename_list):
            files_trn = [self.filename_list[i] for i in idx_trn]
            files_tst = [self.filename_list[i] for i in idx_tst]

            # initialize min median error
            best_score = 0 # TODO: think about what this number should be
            best_params = None
            # TODO: possibly have to write a hyperparam loop or GridSearch
            for this_param in hyperparam_grid:
                this_param_scores = []
                self.pipe.set_params(**this_param)
                for inner_idx_trn, inner_idx_tst in kfold.split(files_trn):
                    inner_files_trn = [files_trn[i] for i in inner_idx_trn]
                    inner_files_tst = [files_trn[i] for i in inner_idx_tst]

                    inner_X_trn, inner_y_trn = self.stack_X_and_y(inner_files_trn)
                    inner_X_tst, inner_y_tst = self.stack_X_and_y(inner_files_tst)
                    self.pipe.fit(inner_X_trn, inner_y_trn)
                    # TODO: make other metrics abailable, maybe
                    this_inner_fold_score = roc_auc_score(inner_y_tst, self.pipe.predict(inner_X_tst))
                    print(this_inner_fold_score)
                    this_param_scores.append(this_inner_fold_score)
                median_score = median(this_param_scores)
                if median_score > best_score:
                    best_params = this_param

            pipe.set_params(**best_params)
            X_trn, y_trn = self.stack_X_and_y(files_trn)
            X_tst, y_tst = self.stack_X_and_y(files_tst)
            pipe.fit(X_trn, y_trn)
            this_fold_score = roc_auc_score(y_tst, self.pipe.predict(X_tst))
            scores.append(this_fold_score)
        print(scores)

    def prepare_features_and_labels(self, features_dir):
        start = time.time()
        self.features_dict = dict.fromkeys(self.filename_list)
        self.labels_dict = dict.fromkeys(self.filename_list)

        pool = mp.Pool(processes=self.n_jobs)
        extractions = [pool.apply_async(self.feat_extractor.get_features_and_labels, args=(filename,))
                       for filename in self.filename_list]

        for e in extractions:
            item = e.get()
            self.features_dict[item[0]] = item[1]
            self.labels_dict[item[0]] = item[2]

        np.savez(os.path.join(features_dir, 'features.npz'), **self.features_dict)
        np.savez(os.path.join(features_dir, 'labels.npz'), **self.labels_dict)
        stop = time.time()
        print("Features are prepared in {} seconds".format(stop-start))

    # TODO: delete debug block
    # def debug_prepare_feats_and_labels(self, features_dir):
    #     self.features_dict = dict.fromkeys(self.filename_list)
    #     self.labels_dict = dict.fromkeys(self.filename_list)
    #     for filename in self.filename_list:
    #         item = self.feat_extractor.get_features_and_labels(filename)
    #         self.features_dict[item[0]] = item[1]
    #         self.labels_dict[item[0]] = item[2]

    def load_features(self, features_dir):
        features_load = np.load(os.path.join(features_dir, 'features.npz'))
        self.features_dict = {file: features_load[file] for file in features_load.files}
        labels_load = np.load(os.path.join(features_dir, 'labels.npz'))
        self.labels_dict = {file: labels_load[file] for file in labels_load.files}

    def stack_X(self, files_list):
        return np.vstack([item for item in [self.features_dict[key] for key in files_list]])

    def stack_y(self, files_list):
        return np.hstack([item for item in [self.labels_dict[key] for key in files_list]])

    def stack_X_and_y(self, files_list):
        return self.stack_X(files_list) , self.stack_y(files_list)
