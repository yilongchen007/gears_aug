import torch
import numpy as np
import scanpy as sc

from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

def evaluate(loader, model, uncertainty, device):
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    # pred_de = []
    # truth_de = []
    results = {}
    logvar = []
    
    for itr, batch in enumerate(loader):

        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            if uncertainty:
                p, unc = model(batch)
                logvar.extend(unc.cpu())
            else:
                p = model(batch)
            t = batch.y.reshape(p.shape)
            pred.extend(p.cpu())
            truth.extend(t.cpu())
            
            # # Differentially expressed genes
            # for itr, de_idx in enumerate(batch.de_idx):
            #     pred_de.append(p[itr, de_idx])
            #     truth_de.append(t[itr, de_idx])

    # all genes
    results['pert_cat'] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results['pred']= pred.detach().cpu().numpy()
    results['truth']= truth.detach().cpu().numpy()

    # pred_de = torch.stack(pred_de)
    # truth_de = torch.stack(truth_de)
    # results['pred_de']= pred_de.detach().cpu().numpy()
    # results['truth_de']= truth_de.detach().cpu().numpy()
    
    if uncertainty:
        results['logvar'] = torch.stack(logvar).detach().cpu().numpy()
    
    return results


def compute_metrics(results):
    """
    Given results from a model run and the ground truth, compute metrics

    """
    metrics = {}
    metrics_pert = {}

    metric2fct = {
           'mse': mse,
           'pearson': pearsonr
    }
    
    for m in metric2fct.keys():
        metrics[m] = []
        # metrics[m + '_de'] = []

    for pert in np.unique(results['pert_cat']):

        metrics_pert[pert] = {}
        p_idx = np.where(results['pert_cat'] == pert)[0]
            
        for m, fct in metric2fct.items():
            if m == 'pearson':
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))[0]
                if np.isnan(val):
                    val = 0
            else:
                val = fct(results['pred'][p_idx].mean(0), results['truth'][p_idx].mean(0))

            metrics_pert[pert][m] = val
            metrics[m].append(metrics_pert[pert][m])

       
        # if pert != 'control':
            
        #     for m, fct in metric2fct.items():
        #         if m == 'pearson':
        #             val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))[0]
        #             if np.isnan(val):
        #                 val = 0
        #         else:
        #             val = fct(results['pred_de'][p_idx].mean(0), results['truth_de'][p_idx].mean(0))
                    
        #         metrics_pert[pert][m + '_de'] = val
        #         metrics[m + '_de'].append(metrics_pert[pert][m + '_de'])

        # else:
        #     for m, fct in metric2fct.items():
        #         metrics_pert[pert][m + '_de'] = 0
    
    for m in metric2fct.keys():
        
        metrics[m] = np.mean(metrics[m])
    #     metrics[m + '_de'] = np.mean(metrics[m + '_de'])
    
    return metrics, metrics_pert


