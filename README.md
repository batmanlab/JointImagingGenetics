# GENUS Exploratory Analysis Repo
## This repo reflects the primary explorartory analysis on the GENUS dataset

#### Library requirements
* Python 2.7
* Matlab 201x
* R 3.3.x
* GCC 4.7.x

#### Preprocessing
* Preprocessing is done on a "base" datafile which contains the concatenated Freesurfer phenotypes and cognitive domain scores. That dataset is much larger than the ones used for classification. <b>preprocess.py</b> in the <b>fs_cog/pred_diag</b> directory is the preprocessing script. 

#### Analysis
* The main results for predicting diagnosis are under the fs_cog directory. All results used an intersection of the freesurfer phenotypes and the 3 domain cognitive score(SOP, RPS, VLM) data. 9 "models" were considered for 2 classifiers - vanilla logistic regression with a sparsity inducing penalty (this is the linear classifier) and a fuzzy-trees classifier (non-linear classifier). The classifiers were chosen for the ability to 1.) reduce the dimension of the input data thus creating "interpretable results" and 2.) predict using the reduced matrix.

The models considered are:

  |Model Variable| Features
  |--------------|:--------------
  |XB            | Freesurfer phenotypes
  |XC            | Cognitive scores
  |XBC           | Freesurfer phenotypes & cognitive scores
  |XBA           | Freesurfer phenotypes & cognitive scores & covariates
  |XBCOV         | Freesurfer phenotypes & covariates
  |XCC           | Cognitive ccores & covariates
  |XBCR          | Freesurfer phenotypes with covariates projected out
  |XCCR          | Cognitive Scores with covariates projected out
  |XBCCR         | Freesurfer phenotypes & cognitive scores with covariates projected out 


### How the classification analysis are carried out
In the <b>fs_cog/pred_diag</b> directory there is a <b>submit.py</b> file that uses two classes one for logistic regression and another for fuzzy/extra trees. These live in the <b>Mods.py</b> file which in turn lives in a module <b>custom</b> in the anaconda environment created for this project. The two classifiers inherit from a base model that simply does some minimal checks on the data, here's the logistic regression classifier, the fuzzy/extra trees is similar though not exact. One other thing to note, this is the "workhorse" of the file for logistic regression - <b>BaseModel</b> and <b>FuzzyForest</b> are also defined in Mods.py: 

```python
class Logistic(BaseModel):
    def fit_model(self, X, y):
        """
        X::pd.DataFrame: Input data
        y::np.ndarray: response for input data
        """
        cv_out = StratifiedShuffleSplit(n_splits=400)
        cv_in = StratifiedKFold(n_splits=5)
        clf = Pipeline([('scaler', StandardScaler()),
                        ('lg', linear_model.LogisticRegressionCV(
                                  penalty='l1',
                                  solver='liblinear',
                                  cv=cv_in))])

        self.res = {'coef':[], 'auc':[], 'model':0}

        for idx, (train, test) in enumerate(cv_out.split(X, y)):
            clf.fit(X[train], y[train])
            prediction = clf.predict(X[test])
            self.res['coef'].append((idx, clf.named_steps['lg'].coef_[0]))
            self.res['auc'].append((idx, roc_auc_score(y[test], prediction)))
        
        self.res['model'] = clf
        output_saved = self.save_pickle(self.res, self.out)
        return output_saved
```

## Variational Bayes
There's a notebook inside <b>bayes/notebooks</b> called <b>variational_bayes</b> that holds both parts of this analyses. It starts from two hdf5 files, one containing sMRI data and another containing SNP data, they are already sorted by rows so that they ccorrespond. Visualizations for the outputs will be part of <b>.py</b> files since one of them requires another anaconda environment which does not have ipython or jupyter installed (and should not). That environment is specialized for the PySurfer module and because of Mayavi - it should not be altered in it's current state. 

## CCA
Sparse CCA is carried out in R, using the package PMA. To do this we use a singularity image just for R, the image can be found in the "singularity" repository.
