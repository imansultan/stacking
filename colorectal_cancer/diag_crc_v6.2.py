# import packages
import os
import math
import time
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

from pingouin import mwu,multicomp

from sklearnex import patch_sklearn
patch_sklearn()

from sklearn.model_selection import StratifiedKFold,GridSearchCV,train_test_split,RepeatedStratifiedKFold,cross_val_score,cross_validate
from sklearn.preprocessing import power_transform,label_binarize,OrdinalEncoder,PowerTransformer
from sklearn.metrics import PrecisionRecallDisplay,RocCurveDisplay,average_precision_score,matthews_corrcoef
from sklearn.dummy import DummyClassifier

from sklearn.linear_model import LogisticRegressionCV,LogisticRegression,SGDClassifier
from sklearn.feature_selection import VarianceThreshold,SelectFpr,SelectFdr,SelectFwe,chi2,f_classif

import missingno as msno

from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,HistGradientBoostingClassifier,VotingClassifier
from xgboost import XGBClassifier

from sklearn.ensemble import StackingClassifier
from mlxtend.feature_selection import ColumnSelector

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# User-defined functions block
def raise_error(error_msg):
    exit('ERROR!: '+error_msg)
    
# evaluate a give model using cross-validation
def evaluate_model(model, X, y, cv_fold, num_repeats):
	cv = RepeatedStratifiedKFold(n_splits=cv_fold, n_repeats=num_repeats, random_state=42)
	scores = cross_validate(model, X, y,
                            scoring='average_precision',
                            cv=cv,
                            n_jobs=-1,
                            error_score='raise',
                            return_train_score=True)
	return scores

# Main function block
def main():
    # Code parameters, will be used later
    print('Script started')
    startTime = time.time()
    np.random.seed(256)
    
    plt.close('all')
    plt.style.use('seaborn-ticks')
    sns.set_context('poster',
                    font_scale=1.0)
    sns.set_style("ticks")
    sns.set_palette('deep')
        
    directory_save='output_data/'
    directory_fig_save=directory_save+'figures/'
    directory_data_save=directory_save+'datasets/'
    
    cv_fold_num=5
    num_cv_repeats=3
    test_set_ratio=0.2
    var_thresh=1
    num_perm_repeats=5
    dev_mode=False
    
    print('Data loading')
    dataset_df=pd.read_excel('input_data/sample_meta_data_crc.xlsx',
                              index_col='SampleID')
    # exclude useless or label leakage columns
    dataset_df.drop(columns=['Site',
                             'Location',
                             'Ethnic',
                             'fit_result',
                             'Hx_Prev',
                             'Hx_Fam_CRC',
                             'Hx_of_Polyps',
                             'stage'],
                    inplace=True)
    
    dir_fig_preproc=directory_fig_save+'preprocessing/'
    if not os.path.exists(dir_fig_preproc):
        os.makedirs(dir_fig_preproc)
    
    # plot diagnosis counts
    plt.figure(figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
    sns.countplot(x='dx',
                  data=dataset_df)
    sns.despine()
    plt.savefig(dir_fig_preproc+'dx_counts.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    # plot extended diagnosis counts
    plt.figure(figsize=(16, 9), dpi=300, facecolor='w', edgecolor='k')
    sns.countplot(x='Dx_Bin',
                  data=dataset_df)
    sns.despine()
    plt.savefig(dir_fig_preproc+'dx_bin_counts.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    # rename label column
    dataset_df.rename(columns={'dx': 'diagnosis'},
                      inplace=True)
    
    # include only interested lables by excluding others
    dataset_df=dataset_df.query("diagnosis != 'adenoma'")
    
    # exclude addiional subcategory label column
    dataset_df.drop(columns=['Dx_Bin'],
                    inplace=True)
    
    # convert string gender to numeric gender
    dataset_df.Gender = pd.Categorical(dataset_df.Gender)
    dataset_df['Sex'] = dataset_df.Gender.cat.codes
    dataset_df['Sex'] = dataset_df['Sex'].astype(int)
    dataset_df.drop(columns=['Gender'],
                    inplace=True)
    
    # reverse one-hot-encoding of ethnicities
    ethnicity_ohe_df=dataset_df.loc[:,['White','Native','Black','Pacific','Asian','Other']]
    dataset_df.drop(columns=['White','Native','Black','Pacific','Asian','Other'],
                    inplace=True)
    ethnicity_df=ethnicity_ohe_df.idxmax(axis=1).to_frame()
    ethnicity_df.rename(columns = {0:'ethnicity_val'},
                        inplace=True)
    ethnicity_df.ethnicity_val = pd.Categorical(ethnicity_df.ethnicity_val)
    ethnicity_df['Ethnicity'] = ethnicity_df.ethnicity_val.cat.codes
    ethnicity_df['Ethnicity'] = ethnicity_df['Ethnicity'].astype(int)
    ethnicity_df.drop(columns=['ethnicity_val'],
                      inplace=True)
    dataset_df=pd.concat([dataset_df,ethnicity_df],
                         axis=1,
                         join='inner')
    
    # calculate BMI
    dataset_df.dropna(subset=['Height','Weight'],
                      how='all',
                      inplace=True)
    dataset_df['BMI'] = np.round(dataset_df['Weight'] / dataset_df['Height'].div(100).pow(2),1)
    dataset_df.drop(columns=['Weight',
                             'Height'],
                    inplace=True)
    
    # # manually impute some columns                
    # dataset_df.fillna({"Hx_Prev":0,
    #                    "Hx_of_Polyps":0,
    #                    "Smoke":0,
    #                    "NSAID":0},
    #                   inplace=True)
            
    # put age as last column
    dataset_df = dataset_df.reindex(columns = [col for col in dataset_df.columns if col != 'Age'] + ['Age'])
    
    # create patient data datframe
    patient_data_df=dataset_df

    otu_df = pd.read_csv('input_data/otu_crc.txt',
                         index_col='SampleID',
                         sep="\t")
    
    if dev_mode:
        otu_df = otu_df.iloc[:,::10]
    
    tax_df = pd.read_csv('input_data/tax_crc.txt',
                         index_col='FeatureID',
                         sep="\t")
    tax_df.fillna("Unknown",inplace=True)
        
    # plot age histogram
    plt.figure(figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
    sns.histplot(data=patient_data_df,
                  x="Age",
                  hue="diagnosis",
                  hue_order=['normal','cancer'],
                  kde=True)
    sns.despine()
    plt.savefig(dir_fig_preproc+'hist_age_diag.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    # plot BMI histogram
    plt.figure(figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
    sns.histplot(data=patient_data_df,
                 x="BMI",
                 hue="diagnosis",
                 hue_order=['normal','cancer'],
                 kde=True)
    sns.despine()
    plt.savefig(dir_fig_preproc+'hist_bmi_diag.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    X_y_df=pd.concat([patient_data_df,otu_df.astype(int)],
                      axis=1,
                      join='inner')
    X_y_df.replace('nan', np.NaN, inplace=True)
    # remove all constant columns
    X_y_df=X_y_df.loc[:, (X_y_df != X_y_df.iloc[0]).any()]
    
    # plot label counts
    plt.figure(figsize=(8, 8), dpi=300, facecolor='w', edgecolor='k')
    sns.countplot(x='diagnosis',
                  order=['normal','cancer'],
                  data=X_y_df)
    sns.despine()
    plt.savefig(dir_fig_preproc+'y_counts.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    # split data to X and y
    y_df=X_y_df['diagnosis'].to_frame()
    X_df=X_y_df.drop(columns=['diagnosis'],
                     inplace=False)
    
    # checking sizes of datasets and possible reduction stratgeies
    num_columns=len(X_df.columns)
    clinical_data_df=X_df.iloc[:,range(0,9)]
    microbiome_data_df=X_df.iloc[:,range(9,num_columns)]
    cat_feat_indices=list(np.arange(7))
    X_cat_df=clinical_data_df.iloc[:,cat_feat_indices]
    
    plt.figure(figsize=(16, 9), dpi=300, facecolor='w', edgecolor='k')
    msno.matrix(clinical_data_df)
    plt.savefig(dir_fig_preproc+'clinical_data_df_missing.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    num_missing_clinical_df=clinical_data_df.isnull().sum().sum()
    clinical_data_df_col_names=list(clinical_data_df.columns)
    
    selector = VarianceThreshold(threshold=var_thresh)
    X_microbiome=microbiome_data_df.to_numpy()
    X_red=selector.fit_transform(X_microbiome)
    X_fpr_new=np.round(SelectFpr(score_func=f_classif).fit_transform(power_transform(X_red),
                                    label_binarize(y_df.to_numpy(),classes=['normal','cancer'])),2)
    X_fdr_new=np.round(SelectFdr(score_func=f_classif).fit_transform(power_transform(X_red),
                                    label_binarize(y_df.to_numpy(),classes=['normal','cancer'])),2)
    X_fwe_new=np.round(SelectFwe(score_func=f_classif).fit_transform(power_transform(X_red),
                                    label_binarize(y_df.to_numpy(),classes=['normal','cancer'])),2)
    
    num_linear_processor=make_pipeline(VarianceThreshold(threshold=var_thresh),
                                       PowerTransformer(method='yeo-johnson', standardize=True),
                                       SelectFpr(score_func=f_classif))
    num_selector = list(range(9,num_columns))
    linear_preprocessor = make_column_transformer((num_linear_processor, num_selector))
    
    X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(X_df,y_df,
                                                                    test_size=test_set_ratio,
                                                                    random_state=128,
                                                                    shuffle=True,
                                                                    stratify=y_df)
    X_train_df['ExampleID']=y_train_df.index
    X_train_df.set_index('ExampleID',inplace=True)
    X_train_df.columns=X_df.columns
    X_test_df.columns=X_df.columns
    X_y_train_df=pd.concat([X_train_df, y_train_df],
                           axis=1,
                           join='inner')
    X_y_test_df=pd.concat([X_test_df, y_test_df],
                           axis=1,
                           join='inner')
    
    sns.set_context('poster',
                    font_scale=1.0)
    sns.set_style("ticks")
    sns.set_palette('deep')
    
    unique, counts = np.unique(y_train_df.to_numpy(), return_counts=True)
    pos_neg_dict=dict(zip(unique, counts))
    scale_pos_weight_value=round(pos_neg_dict['normal']/pos_neg_dict['cancer'],2)
    
    print('Hyperparameter optimization')
    
    estimators = [
                    (
                     'HGBC_clin',
                     make_pipeline(ColumnSelector(range(0,9)),
                                   HistGradientBoostingClassifier(loss='auto', 
                                                                  learning_rate=0.1,
                                                                  max_iter=100,
                                                                  max_leaf_nodes=31,
                                                                  max_depth=None,
                                                                  min_samples_leaf=20,
                                                                  l2_regularization=0.1,
                                                                  max_bins=255,
                                                                  categorical_features=cat_feat_indices,
                                                                  monotonic_cst=None,
                                                                  warm_start=True,
                                                                  early_stopping=True,
                                                                  scoring='average_precision',
                                                                  validation_fraction=0.1,
                                                                  n_iter_no_change=10,
                                                                  tol=1e-07,
                                                                  verbose=0,
                                                                  random_state=42)
                                   )
                    ),
                    (
                     'SGD_LL',
                     make_pipeline(linear_preprocessor,
                                   SGDClassifier(loss='log',
                                                 penalty='elasticnet',
                                                 alpha=0.1,
                                                 l1_ratio=0.5,
                                                 fit_intercept=True,
                                                 max_iter=1000,
                                                 tol=0.001,
                                                 shuffle=True,
                                                 verbose=1,
                                                 epsilon=0.1,
                                                 n_jobs=-1,
                                                 random_state=42,
                                                 learning_rate='optimal',
                                                 eta0=0.0,
                                                 power_t=0.5,
                                                 early_stopping=True,
                                                 validation_fraction=0.1,
                                                 n_iter_no_change=100,
                                                 class_weight='balanced',
                                                 warm_start=False,
                                                 average=False)
                                   )
                    ),
                    (
                      'SGD_HL',
                      make_pipeline(linear_preprocessor,
                                    SGDClassifier(loss='modified_huber',
                                                  penalty='elasticnet',
                                                  alpha=0.1,
                                                  l1_ratio=0.5,
                                                  fit_intercept=True,
                                                  max_iter=1000,
                                                  tol=0.001,
                                                  shuffle=True,
                                                  verbose=1,
                                                  epsilon=0.1,
                                                  n_jobs=-1,
                                                  random_state=42,
                                                  learning_rate='optimal',
                                                  eta0=0.0,
                                                  power_t=0.5,
                                                  early_stopping=True,
                                                  validation_fraction=0.1,
                                                  n_iter_no_change=100,
                                                  class_weight='balanced',
                                                  warm_start=False,
                                                  average=False)
                                    )
                    ),
                    (
                      'MLP',
                      make_pipeline(linear_preprocessor,
                                    MLPClassifier(hidden_layer_sizes=(int(0.25*np.size(X_fpr_new, 1)),),
                                                  activation="tanh",
                                                  solver="adam",
                                                  alpha=1,
                                                  learning_rate_init=0.001,
                                                  max_iter=100,
                                                  shuffle=True,
                                                  random_state=42,
                                                  tol=1e-4,
                                                  early_stopping=True,
                                                  validation_fraction=0.10,
                                                  beta_1=0.9,
                                                  beta_2=0.999,
                                                  verbose=True,
                                                  warm_start=True,
                                                  epsilon=1e-08,
                                                  n_iter_no_change=20)
                                    )
                    ),
                    (
                      'QDA',
                      make_pipeline(linear_preprocessor,
                                    QuadraticDiscriminantAnalysis(priors=None,
                                                                  reg_param=0.5,
                                                                  store_covariance=False,
                                                                  tol=0.0001)
                                    )
                    ),
                    (
                      'KNN',
                      make_pipeline(linear_preprocessor,
                                    KNeighborsClassifier(n_neighbors=10, 
                                                         weights='uniform',
                                                         algorithm='auto',
                                                         leaf_size=30,
                                                         p=2,
                                                         metric='minkowski',
                                                         metric_params=None,
                                                         n_jobs=-1)
                                    )
                    ),
                    (
                      'RF',
                      make_pipeline(linear_preprocessor,
                                    RandomForestClassifier(n_estimators=20, 
                                                           criterion='entropy',
                                                           max_depth=5,
                                                           min_samples_split=2,
                                                           min_samples_leaf=1,
                                                           min_weight_fraction_leaf=0.0,
                                                           max_features=1,
                                                           max_leaf_nodes=None,
                                                           min_impurity_decrease=0.0,
                                                           bootstrap=True,
                                                           oob_score=True,
                                                           n_jobs=-1,
                                                           random_state=42,
                                                           verbose=0,
                                                           warm_start=False,
                                                           class_weight='balanced_subsample',
                                                           ccp_alpha=0.0,
                                                           max_samples=0.6)
                                    )
                    ),
                    (
                      'HGBC_otu',
                      make_pipeline(linear_preprocessor,
                                    HistGradientBoostingClassifier(loss='auto', 
                                                                   learning_rate=0.1,
                                                                   max_iter=100,
                                                                   max_leaf_nodes=10,
                                                                   max_depth=5,
                                                                   min_samples_leaf=5,
                                                                   l2_regularization=1,
                                                                   max_bins=255,
                                                                   categorical_features=None,
                                                                   monotonic_cst=None,
                                                                   warm_start=False,
                                                                   early_stopping=True,
                                                                   scoring='average_precision',
                                                                   validation_fraction=0.1,
                                                                   n_iter_no_change=10,
                                                                   tol=1e-07,
                                                                   verbose=0,
                                                                   random_state=42)
                                    )
                    )
                    ]
    estimator_names= [estimator_val[0] for estimator_val in estimators]
    
    final_estimator = LogisticRegressionCV(Cs=10,
                                           fit_intercept=True,
                                           cv=cv_fold_num,
                                           dual=False,
                                           penalty='elasticnet',
                                           scoring='average_precision',
                                           solver='saga',
                                           tol=0.0001,
                                           max_iter=100,
                                           class_weight='balanced',
                                           n_jobs=-1,
                                           verbose=0,
                                           refit=True,
                                           intercept_scaling=1.0,
                                           multi_class='auto',
                                           random_state=42,
                                           l1_ratios=list(np.round(np.linspace(start=0.1,
                                                                               stop=0.9,
                                                                               num=9,
                                                                               endpoint=True),2)))
    
    classifier=StackingClassifier(estimators=estimators,
                                  final_estimator=final_estimator,
                                  cv=cv_fold_num,
                                  stack_method='auto',
                                  n_jobs=-1,
                                  passthrough=False,
                                  verbose=1)
    
    X_train=X_train_df.to_numpy()
    X_test=X_test_df.to_numpy()
    
    y_train=label_binarize(y_train_df.to_numpy(),classes=['normal','cancer'])
    y_test=label_binarize(y_test_df.to_numpy(),classes=['normal','cancer'])

    classifier.fit(X_train,y_train.ravel())
    named_estimatorts_bunch=classifier.named_estimators_
    super_learner=classifier.final_estimator_
    
    weights=np.round(super_learner.coef_,2)
    logit_weights_df=pd.DataFrame(data=np.transpose(weights),
                              columns=['weight'])
    logit_weights_df['Base_learner']=estimator_names
    logit_weights_df['weight_abs']=logit_weights_df['weight'].abs()
    logit_weights_df['weight_normalized']=logit_weights_df['weight']/logit_weights_df['weight_abs'].max()
    logit_weights_df['Feat_imp']=logit_weights_df['weight_abs']/logit_weights_df['weight_abs'].max()

    logit_weights_df.sort_values(by=['weight_normalized'],
                                 ascending=False,
                                 inplace=True)
    logit_weights_df=logit_weights_df.round({'weight_normalized': 2,
                                             'Feat_imp': 2})
    
    dir_fig_logit=directory_fig_save+'logit/'
    if not os.path.exists(dir_fig_logit):
        os.makedirs(dir_fig_logit)
    
    plt.figure(dpi=300, facecolor='w', edgecolor='k', figsize=(9,16))
    ax = sns.barplot(x="weight_normalized",
                     y="Base_learner",
                     data=logit_weights_df,
                     palette="Blues_r")
    plt.axvline(x=0.0,
                linewidth=3,
                color='k') 
    plt.title("Sorted normalized weights")
    sns.despine()
    plt.savefig(dir_fig_logit+'feat_imp_sorted_superlearner.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    fitted_estimators_dict=dict()
    for estimator_name_val in estimator_names:
        fitted_estimators_dict[estimator_name_val]=named_estimatorts_bunch[estimator_name_val]
    fitted_estimators_dict['Stacked']=classifier
    
    dummy_clf = DummyClassifier(strategy="prior")
    dummy_clf.fit(X_train, y_train.ravel())
    
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.figure(figsize=(12, 12), dpi=300, facecolor='w', edgecolor='k')
    train_disp = PrecisionRecallDisplay.from_estimator(classifier,
                                                       X_train,
                                                       y_train,
                                                       name='Training')
    ax = plt.gca()
    test_disp = PrecisionRecallDisplay.from_estimator(classifier,
                                                      X_test,
                                                      y_test,
                                                      alpha=0.8,
                                                      name='Test',
                                                      ax=ax)
    dummy_disp = PrecisionRecallDisplay.from_estimator(dummy_clf,
                                                       X_train,
                                                       y_train,
                                                       alpha=0.8,
                                                       name='Random',
                                                       ax=ax)
    plt.axhline(y=0.0,
                linewidth=3,
                color='k')    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title('Stacking')
    plt.xlim([0, 1])
    plt.ylim([-0.3, 1])
    plt.legend(loc='lower left')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(dir_fig_logit+'pr_curve_stacking.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.figure(figsize=(12, 12), dpi=300, facecolor='w', edgecolor='k')
    dummy_disp = PrecisionRecallDisplay.from_estimator(dummy_clf,
                                                       X_train,
                                                       y_train,
                                                       name='Random')
    ax = plt.gca()
    list_base_classifiers=list()
    for name, model in fitted_estimators_dict.items():
        print(('Evaluation of %s model' %(name)))
        if name=='Stacked':
            linewidth_val=5.0
        else:
            linewidth_val=1.0
            list_base_classifiers.append((name,model))
        train_disp = PrecisionRecallDisplay.from_estimator(model,
                                                           X_train,
                                                           y_train,
                                                           alpha=0.8,
                                                           linewidth=linewidth_val,
                                                           name=name,
                                                           ax=ax)
    vote_clf=VotingClassifier(estimators=list_base_classifiers,
                              voting='soft',
                              weights=None,
                              n_jobs=-1)
    vote_clf.fit(X_train,
                 y_train)
    avg_disp = PrecisionRecallDisplay.from_estimator(vote_clf,
                                                     X_train,
                                                     y_train,
                                                     alpha=0.8,
                                                     linewidth=3,
                                                     name='SoftVote',
                                                     linestyle='dashdot',
                                                     ax=ax)
    plt.axhline(y=0.0,
                linewidth=3,
                color='k')    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title('Base estimators, training set')
    plt.xlim([0, 1])
    plt.ylim([-1.25, 1])
    plt.legend(loc='lower left')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(dir_fig_logit+'pr_curve_base_training.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.rcParams['figure.figsize'] = [12, 12]
    plt.figure(figsize=(12, 12), dpi=300, facecolor='w', edgecolor='k')
    dummy_disp = PrecisionRecallDisplay.from_estimator(dummy_clf,
                                                       X_test,
                                                       y_test,
                                                       name='Random')
    ax = plt.gca()
    list_base_classifiers=list()
    for name, model in fitted_estimators_dict.items():
        print(('Evaluation of %s model' %(name)))
        if name=='Stacked':
            linewidth_val=5.0
        else:
            linewidth_val=1.0
            list_base_classifiers.append((name,model))
        train_disp = PrecisionRecallDisplay.from_estimator(model,
                                                           X_test,
                                                           y_test,
                                                           alpha=0.8,
                                                           linewidth=linewidth_val,
                                                           name=name,
                                                           ax=ax)
    vote_clf=VotingClassifier(estimators=list_base_classifiers,
                              voting='soft',
                              weights=None,
                              n_jobs=-1)
    vote_clf.fit(X_train,
                 y_train)
    avg_disp = PrecisionRecallDisplay.from_estimator(vote_clf,
                                                     X_test,
                                                     y_test,
                                                     alpha=0.8,
                                                     linewidth=3,
                                                     linestyle='dashdot',
                                                     name='SoftVote',
                                                     ax=ax)
    plt.axhline(y=0.0,
                linewidth=3,
                color='k')    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.title('Base estimators, test set')
    plt.xlim([0, 1])
    plt.ylim([-1.25, 1])
    plt.legend(loc='lower left')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(dir_fig_logit+'pr_curve_base_test.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    cv_results_train_df_list=list()
    cv_results_val_df_list=list()
    fit_time_df_list=list()
    score_time_df_list=list()
    y_train_df=pd.DataFrame({'y_train':y_train.ravel()})
    y_test_df=pd.DataFrame({'y_test':y_test.ravel()})
    
    for name, model in fitted_estimators_dict.items():
        print(('Evaluation of %s model' %(name)))
        scores = evaluate_model(model, X_train, y_train, cv_fold_num, num_repeats=num_cv_repeats)
        scores_train=scores['train_score']
        scores_val=scores['test_score']
        fit_time=scores['fit_time']
        score_time=scores['score_time']
        
        y_train_df[name]=model.predict(X_train)
        y_test_df[name]=model.predict(X_test)
        
        cv_results_tmp_df=pd.DataFrame(data=scores_train,columns=["AP"])
        cv_results_tmp_df['estimator']=name
        cv_results_tmp_df.reset_index(drop=True, inplace=True)
        cv_results_train_df_list.append(cv_results_tmp_df)
        
        cv_results_tmp_df=pd.DataFrame(data=scores_val,columns=["AP"])
        cv_results_tmp_df['estimator']=name
        cv_results_tmp_df.reset_index(drop=True, inplace=True)
        cv_results_val_df_list.append(cv_results_tmp_df)
        
        cv_results_tmp_df=pd.DataFrame(data=fit_time,columns=["time"])
        cv_results_tmp_df['estimator']=name
        cv_results_tmp_df.reset_index(drop=True, inplace=True)
        fit_time_df_list.append(cv_results_tmp_df)
        
        cv_results_tmp_df=pd.DataFrame(data=score_time,columns=["time"])
        cv_results_tmp_df['estimator']=name
        cv_results_tmp_df.reset_index(drop=True, inplace=True)
        score_time_df_list.append(cv_results_tmp_df)
            
    cv_results_train_df=pd.concat(cv_results_train_df_list,axis=0)
    cv_results_val_df=pd.concat(cv_results_val_df_list,axis=0)
    fit_time_df=pd.concat(fit_time_df_list,axis=0)
    score_time_df=pd.concat(score_time_df_list,axis=0)
    
    cv_results_train_df['Set']='Train'
    cv_results_val_df['Set']='Val'
    fit_time_df['Type']='Training'
    score_time_df['Type']='Prediction'
    
    cv_results_df=pd.concat([cv_results_train_df,cv_results_val_df],axis=0)
    cv_time_df=pd.concat([fit_time_df,score_time_df],axis=0)
    
    mcc_values_train=np.zeros(shape=(len(y_train_df.columns),len(y_train_df.columns)))
    mcc_values_test=np.zeros_like(mcc_values_train)
    for row in range(mcc_values_train.shape[0]):
        for col in range(mcc_values_train.shape[1]):
            mcc_values_train[row,col]=np.round(matthews_corrcoef(y_train_df.iloc[:,row],
                                                                 y_train_df.iloc[:,col]),2)
            mcc_values_test[row,col]=np.round(matthews_corrcoef(y_test_df.iloc[:,row],
                                                                y_test_df.iloc[:,col]),2)
            
    mcc_values_train_df=pd.DataFrame(data=mcc_values_train,
                                     index=y_train_df.columns,
                                     columns=y_train_df.columns)
    mcc_values_test_df=pd.DataFrame(data=mcc_values_test,
                                    index=y_test_df.columns,
                                    columns=y_test_df.columns)
    
    plt.figure(figsize=(7, 7), dpi=300, facecolor='w', edgecolor='k')
    sns.set(color_codes=True)
    g = sns.heatmap(mcc_values_train_df,
                    vmin=-1,
                    vmax=+1,
                    cmap="vlag",
                    linewidths=.5,
                    annot=True)
    g.set_yticklabels(labels=y_train_df.columns, rotation=45)
    g.set_xticklabels(labels=y_train_df.columns, rotation=45)
    plt.title('MCC matrix, Training data predictions')
    plt.savefig(dir_fig_logit+"heatmap_mcc_training.png",
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(7, 7), dpi=300, facecolor='w', edgecolor='k')
    sns.set(color_codes=True)
    g = sns.heatmap(mcc_values_test_df,
                    vmin=-1,
                    vmax=+1,
                    cmap="vlag",
                    linewidths=.5,
                    annot=True)
    g.set_yticklabels(labels=y_test_df.columns, rotation=45)
    g.set_xticklabels(labels=y_test_df.columns, rotation=45)
    plt.title('MCC matrix, Test data predictions')
    plt.savefig(dir_fig_logit+"heatmap_mcc_test.png",
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()    
    
    plt.style.use('seaborn-ticks')
    sns.set_context('poster',
                    font_scale=1.0)
    sns.set_style("ticks")
    sns.set_palette('deep')
    
    ap_random=np.round(average_precision_score(y_true=y_train,
                                               y_score=dummy_clf.predict_proba(X_train)[:,1]),
                  2)
    
    plt.figure(dpi=300, facecolor='w', edgecolor='k', figsize=(18,9))            
    ax2 = sns.boxplot(data=cv_results_df,
                      x='estimator',
                      y='AP',
                      hue='Set',
                      showfliers=False,
                      palette="Set2")
    plt.axhline(y=ap_random,
                linewidth=1,
                color='r')            
    sns.despine()
    # ax2.set(ylabel='AP')
    ax2.set_title('CV scores')
    ax2.set_ylim([0, 1.0])
    plt.savefig(dir_fig_logit+'boxplots_ap_combined.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(dpi=300, facecolor='w', edgecolor='k', figsize=(18,9))            
    ax2 = sns.boxplot(data=fit_time_df,
                      x='estimator',
                      y='time',
                      # hue='Type',
                      showfliers=False,
                      palette="Set2")
    sns.despine()
    # ax2.set(ylabel='AP')
    ax2.set_title('CV training times')
    plt.savefig(dir_fig_logit+'boxplots_time_training.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    plt.figure(dpi=300, facecolor='w', edgecolor='k', figsize=(18,9))            
    ax2 = sns.boxplot(data=score_time_df,
                      x='estimator',
                      y='time',
                      # hue='Type',
                      showfliers=False,
                      palette="Set2")
    sns.despine()
    # ax2.set(ylabel='AP')
    ax2.set_title('CV prediction times')
    plt.savefig(dir_fig_logit+'boxplots_time_prediction.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    print('Permutation feature importance')
    result_test = permutation_importance(classifier,
                                          X_test,
                                          y_test,
                                          scoring='average_precision',
                                          n_repeats=num_perm_repeats,
                                          max_samples=1.0,
                                          random_state=42,
                                          n_jobs=-1)
    
    imp_means=result_test.importances_mean
    imp_sd=result_test.importances_std
    
    imp_means_df=pd.DataFrame(data=np.transpose(imp_means),
                              columns=['imp_mean'])
    imp_means_df['Feat_name']=X_train_df.columns
    imp_means_df['Feat_sd']=imp_sd
    imp_means_df['Feat_imp']=imp_means_df['imp_mean']
    
    imp_means_non_zero_df=imp_means_df.query("Feat_imp>0")
    imp_means_non_zero_df.sort_values(by='Feat_imp', ascending=False, inplace=True)
    
    plt.figure(dpi=300, facecolor='w', edgecolor='k', figsize=(9,16))
    ax = sns.barplot(x="Feat_imp",
                      y="Feat_name",
                      data=imp_means_non_zero_df.head(15),
                      palette="Blues_r")
    plt.title("Sorted importances (test set)")
    sns.despine()
    plt.savefig(dir_fig_logit+'feat_imp_sorted_test.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()

    result_train = permutation_importance(classifier,
                                          X_train,
                                          y_train,
                                          scoring='average_precision',
                                          n_repeats=num_perm_repeats,
                                          max_samples=1.0,
                                          random_state=42,
                                          n_jobs=-1)
    
    imp_means=result_train.importances_mean
    imp_sd=result_train.importances_std
    
    imp_means_df=pd.DataFrame(data=np.transpose(imp_means),
                              columns=['imp_mean'])
    imp_means_df['Feat_name']=X_train_df.columns
    imp_means_df['Feat_sd']=imp_sd
    imp_means_df['Feat_imp']=imp_means_df['imp_mean']
    
    imp_means_non_zero_df=imp_means_df.query("Feat_imp>0")
    imp_means_non_zero_df.sort_values(by='Feat_imp', ascending=False, inplace=True)
    
    plt.figure(dpi=300, facecolor='w', edgecolor='k', figsize=(9,16))
    ax = sns.barplot(x="Feat_imp",
                      y="Feat_name",
                      data=imp_means_non_zero_df.head(15),
                      palette="Blues_r")
    plt.title("Sorted importances (training set)")
    sns.despine()
    plt.savefig(dir_fig_logit+'feat_imp_sorted_train.png',
                format='png',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    plt.close()
    
    # calculate script run time and report it
    endTime = time.time()
    runTime=endTime-startTime
    print(('Runtime: %.2f seconds' %runTime))
    
    # stop point for debugging purpose
    dummy=True

# Call main function
if __name__ == "__main__":
    __spec__ = None
    main()