#%%
import os
import joblib
from shutil import rmtree
import pandas as pd
import numpy as np 
from scipy.io import loadmat
from glob import glob

# Load loaders
from core.loaders import *
from core.solver_s2s import Solver
from core.utils import (get_nested_fold_idx, get_ckpt, cal_metric, cal_statistics, mat2df)

# Load model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from core.models import *

# Others
import mlflow as mf
import coloredlogs, logging
from pathlib import Path
import warnings

#########
from core.utils import remove_outlier, group_annot, reversed_total_group_count, get_divs_per_group, rename_metric, save_result
#########


warnings.filterwarnings("ignore")

coloredlogs.install()
logger = logging.getLogger(__name__)  

#%%

class DelayedEarlyStopping(EarlyStopping):
    """
    For checking the annealing, we implement the custom earlystopping callback. 
    Original Paper check the validation step every 2 epochs.
    Perserving this, we also check the validation step every 2 epochs, but call _run_early_stoppinp_check every epoch to check annealing epochs.
    Therefore, the argument, param_traner.check_val_every_n_epoch is setted 1 manumally.
    """

    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        self.check_for_val = 0

    def _run_early_stopping_check(self, trainer):
        # Only run the check after a certain number of epochs       
        if trainer.current_epoch == self.start_epoch-1:
            trainer.model.annealing = False
            print("######## End Annealing Step ########")

        if trainer.current_epoch >= self.start_epoch-1:
            if trainer.model.config.run_val_every == self.check_for_val:
                super()._run_early_stopping_check(trainer)
                self.check_for_val = 0
            self.check_for_val += 1


class SolverS2l(Solver):
    def _get_model(self, ckpt_path_abs=None):
        model = None
        if not ckpt_path_abs:
            if self.config.exp.model_type == "resnet1d":
                model = Resnet1d(self.config.param_model, config=self.config, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "spectroresnet":
                model = SpectroResnet(self.config.param_model, config=self.config, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "mlpbp":
                model = MLPBP(self.config.param_model, config=self.config, random_state=self.config.exp.random_state)
            elif self.config.exp.model_type == "convtr":
                model = ConvTransformer(self.config.param_model, config=self.config, random_state=self.config.exp.random_state)
            else:
                model = eval(self.config.exp.model_type)(self.config.param_model, random_state=self.config.exp.random_state)
            return model
        else:
            if self.config.exp.model_type == "resnet1d":
                model = Resnet1d.load_from_checkpoint(ckpt_path_abs, config=self.config)
            elif self.config.exp.model_type == "spectroresnet":
                model = SpectroResnet.load_from_checkpoint(ckpt_path_abs, config=self.config)
            elif self.config.exp.model_type == "mlpbp":
                model = MLPBP.load_from_checkpoint(ckpt_path_abs, config=self.config)
            elif self.config.exp.model_type == "convtr":
                model = ConvTransformer.load_from_checkpoint(ckpt_path_abs, config=self.config)
            else:
                model = eval(self.config.exp.model_type).load_from_checkpoint(ckpt_path_abs)
            return model
    
    def get_cv_metrics(self, fold_errors, dm, model, outputs, mode="val"):
        if mode=='val':
            loader = dm.val_dataloader()
        elif mode=='test':
            loader = dm.test_dataloader()

        bp_denorm = loader.dataset.bp_denorm

        #--- Predict
        pred = outputs["pred_bp"].numpy()
        true = outputs["true_bp"].numpy()
        naive =  np.mean(dm.train_dataloader(is_print=False).dataset._target_data, axis=0)
        group = outputs["group"].squeeze().numpy()

        #--- Evaluate
        err_dict = {}
        for i, tar in enumerate(['SP', 'DP']):
            tar_acrny = 'sbp' if tar=='SP' else 'dbp'
            pred_bp = bp_denorm(pred[:,i], self.config, tar)
            true_bp = bp_denorm(true[:,i], self.config, tar)
            naive_bp = bp_denorm(naive[i], self.config, tar)

            # error
            err_dict[tar_acrny] = pred_bp - true_bp
            err_dict[tar_acrny+'_hypo'] = pred_bp[group==0] - true_bp[group==0]
            err_dict[tar_acrny+'_normal'] = pred_bp[group==1] - true_bp[group==1]
            err_dict[tar_acrny+'_prehyper'] = pred_bp[group==2] - true_bp[group==2]
            err_dict[tar_acrny+'_hyper2'] = pred_bp[group==3] - true_bp[group==3]
            err_dict[tar_acrny+'_crisis'] = pred_bp[group==4] - true_bp[group==4]

            fold_errors[f"{mode}_{tar_acrny}_pred"].append(pred_bp)
            fold_errors[f"{mode}_{tar_acrny}_hypo_pred"].append(pred_bp[group==0])
            fold_errors[f"{mode}_{tar_acrny}_normal_pred"].append(pred_bp[group==1])
            fold_errors[f"{mode}_{tar_acrny}_prehyper_pred"].append(pred_bp[group==2])
            fold_errors[f"{mode}_{tar_acrny}_hyper2_pred"].append(pred_bp[group==3])
            fold_errors[f"{mode}_{tar_acrny}_crisis_pred"].append(pred_bp[group==4])
            fold_errors[f"{mode}_{tar_acrny}_label"].append(true_bp)
            fold_errors[f"{mode}_{tar_acrny}_hypo_label"].append(true_bp[group==0])
            fold_errors[f"{mode}_{tar_acrny}_normal_label"].append(true_bp[group==1])
            fold_errors[f"{mode}_{tar_acrny}_prehyper_label"].append(true_bp[group==2])
            fold_errors[f"{mode}_{tar_acrny}_hyper2_label"].append(true_bp[group==3])
            fold_errors[f"{mode}_{tar_acrny}_crisis_label"].append(true_bp[group==4])
            fold_errors[f"{mode}_{tar_acrny}_naive"].append([naive_bp]*len(pred_bp))

        fold_errors[f"{mode}_subject_id"].append(loader.dataset.subjects)
        fold_errors[f"{mode}_record_id"].append(loader.dataset.records)
        
        metrics = cal_metric(err_dict, mode=mode)

        group_sbp = []; group_dbp = []
        for i in ['hypo', 'normal', 'prehyper', 'hyper2', 'crisis']:
            if not np.isnan(metrics[f'{mode}/sbp_{i}_mae']):
                group_sbp.append(metrics[f'{mode}/sbp_{i}_mae'])
                group_dbp.append(metrics[f'{mode}/dbp_{i}_mae'])
        metrics[f'{mode}/sbp_group_mae'] = round(np.mean(group_sbp),3)
        metrics[f'{mode}/dbp_group_mae'] = round(np.mean(group_dbp),3)
        return metrics
            
#%%
    def evaluate(self):
        fold_errors_template = {"subject_id":[], "record_id": [],
                                "sbp_naive":[],  "sbp_pred":[], "sbp_label":[],
                                "dbp_naive":[],  "dbp_pred":[],   "dbp_label":[],
                                "sbp_hypo_pred":[], "dbp_hypo_pred":[],
                                "sbp_normal_pred": [], "dbp_normal_pred": [],
                                "sbp_prehyper_pred": [], "dbp_prehyper_pred": [],
                                "sbp_hyper2_pred": [], "dbp_hyper2_pred": [],
                                "sbp_crisis_pred": [], "dbp_crisis_pred": [],
                                "sbp_hypo_label": [], "dbp_hypo_label": [],
                                "sbp_normal_label": [], "dbp_normal_label": [],
                                "sbp_prehyper_label": [], "dbp_prehyper_label": [],
                                "sbp_hyper2_label": [], "dbp_hyper2_label": [],
                                "sbp_crisis_label": [], "dbp_crisis_label": [],
                                }
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["val","test"]}
        
        #--- Data module
        dm = self._get_loader()

        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]
        
        ###--- Grouping Data
        if self.config.remove_outlier:
            all_split_df = remove_outlier(all_split_df)
        all_split_df = group_annot(all_split_df)

        #--- Nested cv 
        self.config = cal_statistics(self.config, all_split_df)
        
        print(self.config)
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])

            self.config.hijack["reversed_total_group_count"] = reversed_total_group_count(train_df)
            self.config.hijack["div_list"] = get_divs_per_group(train_df, self.config)
            
            dm.setup_kfold(train_df, val_df, test_df)

            #--- Init model
            model = self._get_model()

            #--- for annealing
            if not self.config.annealing:
                model.annealing = False

            early_stop_callback = DelayedEarlyStopping(**dict(self.config.param_early_stop), start_epoch=self.config.annealing) 
            checkpoint_callback = ModelCheckpoint(**dict(self.config.logger.param_ckpt))
            
            lr_logger = LearningRateMonitor()
            trainer = MyTrainer(**dict(self.config.param_trainer), callbacks=[early_stop_callback, checkpoint_callback],
            logger=self.config.pl_log)

            #--- trainer main loop
            mf.pytorch.autolog()
            with mf.start_run(run_name=f"cv{foldIdx}", nested=True) as run:
                # train
                trainer.fit(model, dm)
                print("run_id", run.info.run_id)
                artifact_uri, ckpt_path = get_ckpt(mf.get_run(run_id=run.info.run_id))

                # load best ckpt
                ckpt_path_abs = str(Path(artifact_uri)/ckpt_path[0])
                model = self._get_model(ckpt_path_abs=ckpt_path_abs)
                
                # evaluate
                val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False)
                test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)

                # save updated model
                trainer.model = model
                #trainer.save_checkpoint(ckpt_path_abs) ## saving disk space

                # clear redundant mlflow models (save disk space)
                redundant_model_path = Path(artifact_uri)/'model'
                if redundant_model_path.exists(): rmtree(redundant_model_path)

                metrics = self.get_cv_metrics(fold_errors, dm, model, val_outputs, mode="val")
                metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")

                # Save Fold
                if not self.config.no_result_save:
                    result_path = f"{self.config.exp.data_name}/{self.config.exp.model_type}/{self.config.method})"
                    os.makedirs(f'./{result_path}', exist_ok=True)

                    filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith('nv')}
                    filtered_metrics["name"] = self.config.exp.exp_detail
                    filtered_metrics = rename_metric(filtered_metrics, self.config)
                    
                    save_result(filtered_metrics, path=f'./{result_path}/reseults_fold_{foldIdx}_detail.csv')

                    filtered_metrics = {k: v for k, v in filtered_metrics.items() if not k.endswith('_std')}
                    filtered_metrics = {k: v for k, v in filtered_metrics.items() if not k.endswith('_me')}
                    save_result(filtered_metrics, path=f'./{result_path}/reseults_fold_{foldIdx}.csv')
                logger.info(f"\t {metrics}")
                mf.log_metrics(metrics)

            ### clear mlruns checkpoint for saving disk space
            rmtree(os.path.dirname(str(Path(artifact_uri))))

            #--- Save to model directory
            if self.config.save_model:
                os.makedirs(self.config.path.model_directory, exist_ok=True)
                trainer.save_checkpoint("{}/{}/fold{}-test_sp={:.3f}-test_dp={:.3f}.ckpt".format(
                                                                            self.config.path.model_directory,
                                                                            self.config.exp.exp_name,
                                                                            foldIdx, 
                                                                            metrics["test/sbp_mae"],
                                                                            metrics["test/dbp_mae"]))

        #--- compute final metric
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)
        
        for mode in ['val', 'test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'sbp_hypo', 'sbp_normal', 'sbp_prehyper', 'sbp_hyper2', 'sbp_crisis', 
                        'dbp', 'dbp_hypo', 'dbp_normal', 'dbp_prehyper', 'dbp_hyper2', 'dbp_crisis']}
            tmp_metric = cal_metric(err_dict, mode=mode)
            out_metric.update(tmp_metric)
        
        # Group avg
        group_sbp = []; group_dbp = []
        for i in ['hypo', 'normal', 'prehyper', 'hyper2', 'crisis']:
            if not np.isnan(out_metric[f'{mode}/sbp_{i}_mae']):
                group_sbp.append(out_metric[f'{mode}/sbp_{i}_mae'])
                group_dbp.append(out_metric[f'{mode}/dbp_{i}_mae'])
        out_metric[f'{mode}/sbp_group_mae'] = round(np.mean(group_sbp),3)
        out_metric[f'{mode}/dbp_group_mae'] = round(np.mean(group_dbp),3)
        return out_metric
    
    def test(self):
        results = {}
        fold_errors_template = {"subject_id":[], "record_id": [],
                                "sbp_naive":[],  "sbp_pred":[], "sbp_label":[],
                                "dbp_naive":[],  "dbp_pred":[],   "dbp_label":[],
                                "sbp_hypo_pred":[], "dbp_hypo_pred":[],
                                "sbp_normal_pred": [], "dbp_normal_pred": [],
                                "sbp_prehyper_pred": [], "dbp_prehyper_pred": [],
                                "sbp_hyper2_pred": [], "dbp_hyper2_pred": [],
                                "sbp_crisis_pred": [], "dbp_crisis_pred": [],
                                "sbp_hypo_label": [], "dbp_hypo_label": [],
                                "sbp_normal_label": [], "dbp_normal_label": [],
                                "sbp_prehyper_label": [], "dbp_prehyper_label": [],
                                "sbp_hyper2_label": [], "dbp_hyper2_label": [],
                                "sbp_crisis_label": [], "dbp_crisis_label": [],
                                }
        fold_errors = {f"{mode}_{k}":[] for k,v in fold_errors_template.items() for mode in ["test"]}
        
        #--- Data module
        dm = self._get_loader()
        
        #--- Load data
        if self.config.exp.subject_dict.endswith('.pkl'):
            all_split_df = joblib.load(self.config.exp.subject_dict)
        elif self.config.exp.subject_dict.endswith('fold'):
            all_split_df = [mat2df(loadmat(f"{self.config.exp.subject_dict}_{i}.mat")) for i in range(self.config.exp.N_fold)]
        
        ###--- Grouping Data
        if self.config.remove_outlier:
            all_split_df = remove_outlier(all_split_df)
        all_split_df = group_annot(all_split_df)

        #--- Nested cv 
        self.config = cal_statistics(self.config, all_split_df)
        for foldIdx, (folds_train, folds_val, folds_test) in enumerate(get_nested_fold_idx(self.config.exp.N_fold)):
            if (self.config.exp.cv=='HOO') and (foldIdx==1):  break
            train_df = pd.concat(np.array(all_split_df)[folds_train])
            val_df = pd.concat(np.array(all_split_df)[folds_val])
            test_df = pd.concat(np.array(all_split_df)[folds_test])
            
            dm.setup_kfold(train_df, val_df, test_df)

            #--- load trained model
            if 'param_trainer' in self.config.keys():
                trainer = MyTrainer(**dict(self.config.param_trainer), logger=self.config.pl_log)
            else:
                trainer = MyTrainer(logger=self.config.pl_log)
            ckpt_apth_abs = glob(f'{self.config.param_test.model_path}{foldIdx}' + '*.ckpt')[0]
            model = self._get_model(ckpt_path_abs=ckpt_apth_abs)
            model.eval()
            trainer.model = model
            
            #--- get test output
            val_outputs = trainer.validate(model=model, val_dataloaders=dm.val_dataloader(), verbose=False)
            test_outputs = trainer.test(model=model, test_dataloaders=dm.test_dataloader(), verbose=True)
            metrics = self.get_cv_metrics(fold_errors, dm, model, test_outputs, mode="test")
            logger.info(f"\t {metrics}")
            if not self.config.no_result_save:
                    result_path = f"{self.config.exp.data_name}/{self.config.exp.model_type}/{self.config.method}/test"
                    os.makedirs(f'./{result_path}', exist_ok=True)

                    filtered_metrics = {k: v for k, v in metrics.items() if not k.startswith('nv')}
                    filtered_metrics["name"] = self.config.exp.exp_detail
                    filtered_metrics = rename_metric(filtered_metrics, self.config)
                    
                    save_result(filtered_metrics, path=f'./{result_path}/reseults_fold_{foldIdx}_detail.csv')

                    filtered_metrics = {k: v for k, v in filtered_metrics.items() if not k.endswith('_std')}
                    filtered_metrics = {k: v for k, v in filtered_metrics.items() if not k.endswith('_me')}
                    save_result(filtered_metrics, path=f'./{result_path}/reseults_fold_{foldIdx}.csv')

        #--- compute final metric
        results['fold_errors'] = fold_errors    
        out_metric = {}
        fold_errors = {k:np.concatenate(v, axis=0) for k,v in fold_errors.items()}
        err_dict = {tar: fold_errors[f"test_{tar}_naive"] - fold_errors[f"test_{tar}_label"] \
                    for tar in ['sbp', 'dbp']}
        naive_metric = cal_metric(err_dict, mode='nv')
        out_metric.update(naive_metric)
        
        for mode in ['test']:
            err_dict = {tar: fold_errors[f"{mode}_{tar}_pred"] - fold_errors[f"{mode}_{tar}_label"] \
                        for tar in ['sbp', 'sbp_hypo', 'sbp_normal', 'sbp_prehyper', 'sbp_hyper2', 'sbp_crisis', 
                        'dbp', 'dbp_hypo', 'dbp_normal', 'dbp_prehyper', 'dbp_hyper2', 'dbp_crisis']}
            tmp_metric = cal_metric(err_dict, mode=mode)
            out_metric.update(tmp_metric)
        
        # Group Avg
        group_sbp = []; group_dbp = []
        for i in ['hypo', 'normal', 'prehyper', 'hyper2', 'crisis']:
            if not np.isnan(out_metric[f'{mode}/sbp_{i}_mae']):
                group_sbp.append(out_metric[f'{mode}/sbp_{i}_mae'])
                group_dbp.append(out_metric[f'{mode}/dbp_{i}_mae'])
        out_metric[f'{mode}/sbp_group_mae'] = round(np.mean(group_sbp),3)
        out_metric[f'{mode}/dbp_group_mae'] = round(np.mean(group_dbp),3)

        results['out_metric'] = out_metric 
        os.makedirs(os.path.dirname(self.config.param_test.save_path), exist_ok=True)
        joblib.dump(results, self.config.param_test.save_path)  
        
        
        return out_metric