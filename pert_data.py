from torch_geometric.data import Data
import torch
import numpy as np
import pickle
from torch_geometric.data import DataLoader
import os
import sys
import scanpy as sc
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from .utils import print_sys, dataverse_download, tar_data_download_wrapper, get_genes_from_perts





##################################################################################
#
# Data splitter
#
##################################################################################

def filter_pert_in_go(condition, pert_names):
    """
    Filter perturbations in GO graph

    Args:
        condition (str): perturbation condition
        pert_names (list): list of perturbations
    """

    if condition == 'control':
        return True
    else:
        cond1 = condition.split('+')[0]
        cond2 = 'control' if '+' not in condition else condition.split('+')[1]
        num_ctrl = (cond1 == 'control') + (cond2 == 'control')
        num_in_perts = (cond1 in pert_names) + (cond2 in pert_names)
        if num_ctrl + num_in_perts == 2:
            return True
        else:
            return False



def rank_genes_groups_by_cov(
    adata,
    groupby,
    control_group,
    covariate,
    pool_doses=False,
    n_genes=50,
    rankby_abs=True,
    key_added='rank_genes_groups_cov',
    return_dict=False,
):

    gene_dict = {}
    cov_categories = adata.obs[covariate].unique()
    for cov_cat in cov_categories:
        #name of the control group in the groupby obs column
        control_group_cov = '_'.join([cov_cat, control_group])

        #subset adata to cells belonging to a covariate category
        adata_cov = adata[adata.obs[covariate]==cov_cat]

        #compute DEGs
        sc.tl.rank_genes_groups(
            adata_cov,
            groupby=groupby,
            reference=control_group_cov,
            rankby_abs=rankby_abs,
            n_genes=n_genes,
            use_raw=False
        )

        #add entries to dictionary of gene sets
        de_genes = pd.DataFrame(adata_cov.uns['rank_genes_groups']['names'])
        for group in de_genes:
            gene_dict[group] = de_genes[group].tolist()

    adata.uns[key_added] = gene_dict

    if return_dict:
        return gene_dict

    
def get_sig_genes(adata, skip_calc_de):
    adata.obs.loc[:, 'dose_val'] = adata.obs.condition.apply(lambda x: '1+1' if len(x.split('+')) == 2 else '1')
    adata.obs.loc[:, 'ctrl'] = adata.obs.condition.apply(lambda x: 0 if len(x.split('+')) == 2 else 1)
    adata.obs.loc[:, 'condition_name'] =  adata.obs.apply(lambda x: '_'.join([x.cell_type, x.condition, x.dose_val]), axis = 1) 
    
    adata.obs = adata.obs.astype('category')
    if skip_calc_de:
        return adata

    # rank_genes_groups_by_cov(adata, 
    #                     groupby='condition_name', 
    #                     covariate='cell_type', 
    #                     control_group='ctrl_1', 
    #                     n_genes=len(adata.var),
    #                     key_added = 'rank_genes_groups_cov_all')

    # calculate mean expression for each condition
    unique_conditions = adata.obs['perturbation'].unique()
    conditions2index = {}
    for i in unique_conditions:
        conditions2index[i] = np.where(adata.obs['perturbation'] == i)[0]

    condition2mean_expression = {}
    for i, j in conditions2index.items():
        condition2mean_expression[i] = np.mean(adata.X[j], axis = 0)
    pert_list = np.array(list(condition2mean_expression.keys()))
    mean_expression = np.array(list(condition2mean_expression.values())).reshape(len(adata.obs['perturbation'].unique()), adata.X.toarray().shape[1])
    ctrl = mean_expression[np.where(pert_list == 'control')[0]]
    
    ## in silico modeling and upperbounding
    pert2pert_full_id = dict(adata.obs[['perturbation', 'perturbation_name']].values)
    pert_full_id2pert = dict(adata.obs[['perturbation_name', 'perturbation']].values)

    gene_id2idx = dict(zip(adata.var.index.values, range(len(adata.var))))
    gene_idx2id = dict(zip(range(len(adata.var)), adata.var.index.values))

    non_zeros_gene_idx = {}
    top_non_dropout_de_20 = {}
    top_non_zero_de_20 = {}
    non_dropout_gene_idx = {}

    for pert in adata.uns['rank_genes_groups_cov_all'].keys():
        p = pert_full_id2pert[pert]
        X = np.mean(adata[adata.obs.condition == p].X, axis = 0)

        non_zero = np.where(np.array(X)[0] != 0)[0]
        zero = np.where(np.array(X)[0] == 0)[0]
        true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
        non_dropouts = np.concatenate((non_zero, true_zeros))

        top = adata.uns['rank_genes_groups_cov_all'][pert]
        gene_idx_top = [gene_id2idx[i] for i in top]

        non_dropout_20 = [i for i in gene_idx_top if i in non_dropouts][:20]
        non_dropout_20_gene_id = [gene_idx2id[i] for i in non_dropout_20]

        non_zero_20 = [i for i in gene_idx_top if i in non_zero][:20]
        non_zero_20_gene_id = [gene_idx2id[i] for i in non_zero_20]

        non_zeros_gene_idx[pert] = np.sort(non_zero)
        non_dropout_gene_idx[pert] = np.sort(non_dropouts)
        top_non_dropout_de_20[pert] = np.array(non_dropout_20_gene_id)
        top_non_zero_de_20[pert] = np.array(non_zero_20_gene_id)
        
    non_zero = np.where(np.array(X)[0] != 0)[0]
    zero = np.where(np.array(X)[0] == 0)[0]
    true_zeros = np.intersect1d(zero, np.where(np.array(ctrl)[0] == 0)[0])
    non_dropouts = np.concatenate((non_zero, true_zeros))
    
    adata.uns['top_non_dropout_de_20'] = top_non_dropout_de_20
    adata.uns['non_dropout_gene_idx'] = non_dropout_gene_idx
    adata.uns['non_zeros_gene_idx'] = non_zeros_gene_idx
    adata.uns['top_non_zero_de_20'] = top_non_zero_de_20
    
    return adata



class DataSplitter():
    """
    Class for handling data splitting. This class is able to generate new
    data splits and assign them as a new attribute to the data file.
    """
    def __init__(self, adata, split_type='single', seen=0):
        self.adata = adata
        self.split_type = split_type
        self.seen = seen

    def split_data(self, test_size=0.1, 
        test_pert_genes=None, test_perts=None, train_filter_func = None,
        split_name='split', seed=None, 
        val_size = 0.1, train_gene_set_size = 0.75, combo_seen2_train_frac = 0.75, 
        only_test_set_perts = False):
        """
        Split dataset and adds split as a column to the dataframe
        Note: split categories are train, val, test
        """
        np.random.seed(seed=seed)
        unique_perts = [p for p in self.adata.obs['perturbation'].unique() if
                        p != 'control']
        
        if self.split_type == 'simulation':
            train, test, test_subgroup = self.get_simulation_split(unique_perts,
                                                                  train_gene_set_size,
                                                                  combo_seen2_train_frac, 
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split(train,
                                                                  0.9,
                                                                  0.9,
                                                                  seed)
            ## adding back ctrl to train...
            train.append('control')
        elif self.split_type == 'simulation_single':
            train, test, test_subgroup = self.get_simulation_split_single(unique_perts,
                                                                  train_gene_set_size,
                                                                  seed, test_perts, only_test_set_perts)
            train, val, val_subgroup = self.get_simulation_split_single(train,
                                                                  0.9,
                                                                  seed)
        elif self.split_type == 'no_test':
            train, val = self.get_split_list(unique_perts,
                                          test_size=val_size)      
        else:
            train, test = self.get_split_list(unique_perts,
                                          test_pert_genes=test_pert_genes,
                                          test_perts=test_perts,
                                          test_size=test_size)
            
            train, val = self.get_split_list(train, test_size=val_size)

        map_dict = {x: 'train' for x in train}
        map_dict.update({x: 'val' for x in val})
        if self.split_type != 'no_test':
            map_dict.update({x: 'test' for x in test})

        self.adata.obs[split_name] = self.adata.obs['perturbation'].map(map_dict)
        if train_filter_func is None:
            train_filter_func = lambda x:x['perturbation']=='control'
        self.adata.obs.loc[self.adata.obs.apply(train_filter_func, axis=1), split_name] = 'train'

        if self.split_type == 'simulation':
            return self.adata, {'test_subgroup': test_subgroup, 
                                'val_subgroup': val_subgroup
                               }
        else:
            return self.adata
    
    def get_simulation_split_single(self, pert_list, train_gene_set_size = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        unique_pert_genes = get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)  
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        assert len(unseen_single) + len(pert_single_train) == len(pert_list)
        
        return pert_single_train, unseen_single, {'unseen_single': unseen_single}
    
    def get_simulation_split(self, pert_list, train_gene_set_size = 0.85, combo_seen2_train_frac = 0.85, seed = 1, test_set_perts = None, only_test_set_perts = False):
        
        unique_pert_genes = get_genes_from_perts(pert_list)
        
        pert_train = []
        pert_test = []
        np.random.seed(seed=seed)
        
        if only_test_set_perts and (test_set_perts is not None):
            ood_genes = np.array(test_set_perts)
            train_gene_candidates = np.setdiff1d(unique_pert_genes, ood_genes)
        else:
            ## a pre-specified list of genes
            train_gene_candidates = np.random.choice(unique_pert_genes,
                                                    int(len(unique_pert_genes) * train_gene_set_size), replace = False)

            if test_set_perts is not None:
                num_overlap = len(np.intersect1d(train_gene_candidates, test_set_perts))
                train_gene_candidates = train_gene_candidates[~np.isin(train_gene_candidates, test_set_perts)]
                ood_genes_exclude_test_set = np.setdiff1d(unique_pert_genes, np.union1d(train_gene_candidates, test_set_perts))
                train_set_addition = np.random.choice(ood_genes_exclude_test_set, num_overlap, replace = False)
                train_gene_candidates = np.concatenate((train_gene_candidates, train_set_addition))
                
            ## ood genes
            ood_genes = np.setdiff1d(unique_pert_genes, train_gene_candidates)                
        
        pert_single_train = self.get_perts_from_genes(train_gene_candidates, pert_list,'single')
        pert_combo = self.get_perts_from_genes(train_gene_candidates, pert_list,'combo')
        pert_train.extend(pert_single_train)
        
        ## the combo set with one of them in OOD
        combo_seen1 = [x for x in pert_combo if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 1]
        pert_test.extend(combo_seen1)
        
        pert_combo = np.setdiff1d(pert_combo, combo_seen1)
        ## randomly sample the combo seen 2 as a test set, the rest in training set
        np.random.seed(seed=seed)
        pert_combo_train = np.random.choice(pert_combo, int(len(pert_combo) * combo_seen2_train_frac), replace = False)
       
        combo_seen2 = np.setdiff1d(pert_combo, pert_combo_train).tolist()
        pert_test.extend(combo_seen2)
        pert_train.extend(pert_combo_train)
        
        ## unseen single
        unseen_single = self.get_perts_from_genes(ood_genes, pert_list, 'single')
        combo_ood = self.get_perts_from_genes(ood_genes, pert_list, 'combo')
        pert_test.extend(unseen_single)
        
        ## here only keeps the seen 0, since seen 1 is tackled above
        combo_seen0 = [x for x in combo_ood if len([t for t in x.split('+') if
                                     t in train_gene_candidates]) == 0]
        pert_test.extend(combo_seen0)
        assert len(combo_seen1) + len(combo_seen0) + len(unseen_single) + len(pert_train) + len(combo_seen2) == len(pert_list)

        return pert_train, pert_test, {'combo_seen0': combo_seen0,
                                       'combo_seen1': combo_seen1,
                                       'combo_seen2': combo_seen2,
                                       'unseen_single': unseen_single}
        
    def get_split_list(self, pert_list, test_size=0.1,
                       test_pert_genes=None, test_perts=None,
                       hold_outs=True):
        """
        Splits a given perturbation list into train and test with no shared
        perturbations
        """

        single_perts = [p for p in pert_list if '+' in p and p != 'control']
        combo_perts = [p for p in pert_list if '+' not in p]
        unique_pert_genes = get_genes_from_perts(pert_list)
        hold_out = []

        if test_pert_genes is None:
            test_pert_genes = np.random.choice(unique_pert_genes,
                                        int(len(single_perts) * test_size))

        # Only single unseen genes (in test set)
        # Train contains both single and combos
        if self.split_type == 'single' or self.split_type == 'single_only':
            test_perts = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                   'single')
            if self.split_type == 'single_only':
                # Discard all combos
                hold_out = combo_perts
            else:
                # Discard only those combos which contain test genes
                hold_out = self.get_perts_from_genes(test_pert_genes, pert_list,
                                                     'combo')
        
        elif self.split_type == 'no_test':
            if test_perts is None:
                test_perts = np.random.choice(pert_list,
                                    int(len(pert_list) * test_size))
            

        elif self.split_type == 'combo':
            if self.seen == 0:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 1 gene seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 0]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 1:
                # NOTE: This can reduce the dataset size!
                # To prevent this set 'holdouts' to False, this will cause
                # the test set to have some perturbations with 2 genes seen
                single_perts = self.get_perts_from_genes(test_pert_genes,
                                                         pert_list, 'single')
                combo_perts = self.get_perts_from_genes(test_pert_genes,
                                                        pert_list, 'combo')

                if hold_outs:
                    # This just checks that none of the combos have 2 seen genes
                    hold_out = [t for t in combo_perts if
                                len([t for t in t.split('+') if
                                     t not in test_pert_genes]) > 1]
                combo_perts = [c for c in combo_perts if c not in hold_out]
                test_perts = single_perts + combo_perts

            elif self.seen == 2:
                if test_perts is None:
                    test_perts = np.random.choice(combo_perts,
                                     int(len(combo_perts) * test_size))       
                else:
                    test_perts = np.array(test_perts)
        else:
            if test_perts is None:
                test_perts = np.random.choice(combo_perts,
                                    int(len(combo_perts) * test_size))
        
        train_perts = [p for p in pert_list if (p not in test_perts)
                                        and (p not in hold_out)]
        return train_perts, test_perts

    def get_perts_from_genes(self, genes, pert_list, type_='both'):
        """
        Returns all single/combo/both perturbations that include a gene
        """

        single_perts = [p for p in pert_list if ('+' not in p) and (p != 'control')]
        combo_perts = [p for p in pert_list if '+' in p]
        
        perts = []
        
        if type_ == 'single':
            pert_candidate_list = single_perts
        elif type_ == 'combo':
            pert_candidate_list = combo_perts
        elif type_ == 'both':
            pert_candidate_list = pert_list
            
        for p in pert_candidate_list:
            for g in genes:
                if g in [i for i in p.split('+') if i != 'control']:
                    perts.append(p)
                    break
        return perts


##################################################################################
#
# PertData
#
##################################################################################

class PertData:
    """
    Class for loading and processing perturbation data

    Attributes
    ----------
    data_path: str
        Path to save/load data
    gene_set_path: str
        Path to gene set to use for perturbation graph
    default_pert_graph: bool
        Whether to use default perturbation graph or not
    dataset_name: str
        Name of dataset
    dataset_path: str
        Path to dataset
    adata: AnnData
        AnnData object containing dataset
    dataset_processed: bool
        Whether dataset has been processed or not
    ctrl_adata: AnnData
        AnnData object containing control samples
    gene_names: list
        List of gene names
    node_map: dict
        Dictionary mapping gene names to indices
    split: str
        Split type
    seed: int
        Seed for splitting
    subgroup: str
        Subgroup for splitting
    train_gene_set_size: int
        Number of genes to use for training

    """
    
    def __init__(self, data_path, 
                 gene_set_path=None, 
                 default_pert_graph=True):
        """
        Parameters
        ----------

        data_path: str
            Path to save/load data
        gene_set_path: str
            Path to gene set to use for perturbation graph
        default_pert_graph: bool
            Whether to use default perturbation graph or not

        """

        
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
    
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['perturbation'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path, 'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
    
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
            
    def load(self, dataset_name = None):
        self.dataset_path = os.path.join(self.data_path, dataset_name)
        if os.path.exists(self.dataset_path):
            adata_path = os.path.join(self.dataset_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = dataset_name
        else:
            raise ValueError("data attribute is either norman, adamson, dixit "
                             "replogle_k562 or replogle_rpe1 "
                             "or a path to an h5ad file")
        
        self.set_pert_genes()        
        not_in_go_pert = np.array(self.adata.obs[
                                  self.adata.obs['perturbation'].apply(
                                  lambda x:not filter_pert_in_go(x,
                                        self.pert_names))]['perturbation'].unique())
        print_sys(f'These {len(not_in_go_pert)} perturbations are not in the GO graph and their '
            'perturbation can thus not be predicted')                                        
        print_sys(not_in_go_pert)
        
        filter_go = self.adata.obs[self.adata.obs['perturbation'].apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :]
        
            
    def new_data_process(self, dataset_name,
                         adata = None,
                        #  skip_calc_de = False
                         ):
        """
        Process new dataset

        Parameters
        ----------
        dataset_name: str
            Name of dataset
        adata: AnnData object
            AnnData object containing gene expression data
        # skip_calc_de: bool
        #     If True, skip differential expression calculation

        Returns
        -------
        None

        """
        
        if 'perturbation' not in adata.obs.columns.values:
            raise ValueError("Please specify perturbation")
        if 'gene_name' not in adata.var.columns.values:
            raise ValueError("Please specify gene name")
        if 'celltype' not in adata.obs.columns.values:
            raise ValueError("Please specify cell type")
        
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        
        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder
        self.adata = adata
        # self.adata = get_DE_genes(adata, skip_calc_de)
        # if not skip_calc_de:
        # self.adata = get_sig_genes(self.adata, skip_calc_de)
        self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        
        self.set_pert_genes()
        self.ctrl_adata = self.adata[self.adata.obs['perturbation'] == 'control']
        self.gene_names = self.adata.var.gene_name
        # pyg_path = os.path.join(save_data_folder, 'data_pyg')
        # if not os.path.exists(pyg_path):
        #     os.mkdir(pyg_path)
        # dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        # print_sys("Creating pyg object for each cell in the data...")
        # self.create_dataset_file()
        # print_sys("Saving new dataset pyg object at " + dataset_fname) 
        # pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
        # print_sys("Done!")
        
    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None, only_test_set_perts = False, test_pert_genes = None, train_filter_func=None,
                      split_dict_path=None):

        """
        Prepare splits for training and testing

        Parameters
        ----------
        split: str
            Type of split to use. Currently, we support 'simulation',
            'simulation_single', 'combo_seen0', 'combo_seen1', 'combo_seen2',
            'single', 'no_test', 'no_split', 'custom'
        seed: int
            Random seed
        train_gene_set_size: float
            Fraction of genes to use for training
        combo_seen2_train_frac: float
            Fraction of combo seen2 perturbations to use for training
        combo_single_split_test_set_fraction: float
            Fraction of combo single perturbations to use for testing
        test_perts: list
            List of perturbations to use for testing
        only_test_set_perts: bool
            If True, only use test set perturbations for testing
        test_pert_genes: list
            List of genes to use for testing
        split_dict_path: str
            Path to dictionary used for custom split. Sample format:
                {'train': [X, Y], 'val': [P, Q], 'test': [Z]}

        Returns
        -------
        None

        """
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split', 'custom']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        
        if split == 'custom':
            try:
                with open(split_dict_path, 'rb') as f:
                    self.set2conditions = pickle.load(f)
            except:
                raise ValueError('Please set split_dict_path for custom split')
            return
            
        self.train_gene_set_size = train_gene_set_size
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)
        
        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'
        
        if os.path.exists(split_path):
            print('here1')
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup

            self.adata.obs['split'] = self.adata.obs.apply(
                lambda row: 'train' if row['perturbation'] in set2conditions[row['dataset']]['train'] else
                ('val' if row['perturbation'] in set2conditions[row['dataset']]['val'] else 'test'), axis=1)
        else:
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                # simulation split
                DS = DataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size, 
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts,
                                                train_filter_func = train_filter_func
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
                
            elif split[:5] == 'combo':
                # combo perturbation
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')
                
                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)

            elif split == 'single':
                # single perturbation
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)

            elif split == 'no_test':
                # no test set
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(seed=seed)
            
            elif split == 'no_split':
                # no split
                adata = self.adata
                adata.obs['split'] = 'test'
                 
            # set2conditions = dict(adata.obs.groupby('split').agg({'perturbation':
            #                                             lambda x: x})['perturbation'])
            # set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()}
            set2conditions = {}
            for dataset in adata.obs['dataset'].unique():
                set2conditions[dataset] = {'test': [], 'train': [], 'val': []}
                set2conditions[dataset].update(
                    adata.obs[adata.obs['dataset']==dataset].groupby('split').agg(
                        {'perturbation': lambda x: x.unique().tolist()}
                    )['perturbation'].to_dict()
                )
            pickle.dump(set2conditions, open(split_path, "wb"))

            # self.adata.write_h5ad(os.path.join(self.dataset_path, 'perturb_processed.h5ad'))
            print_sys("Saving new splits at " + split_path)
            
        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))


        pyg_path = os.path.join(self.dataset_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
                
        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed_all = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['perturbation'] == 'control']
            self.gene_names = self.adata.var.gene_name
            
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed_all, open(dataset_fname, "wb"))    
            print_sys("Done!")

        print_sys("Done!")
        
    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        test_batch_size: int
            Batch size for testing

        Returns
        -------
        dict
            Dictionary of dataloaders

        """
        if test_batch_size is None:
            test_batch_size = batch_size
            
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name
        
        print_sys("Creating dataloaders....")
        self.split_dataset_file()
        
        # Set up dataloaders
        train_loader = DataLoader(self.dataset_processed['train'],
                            batch_size=batch_size, shuffle=True, drop_last = True)
        val_loader = DataLoader(self.dataset_processed['val'],
                            batch_size=batch_size, shuffle=True)
        
        if self.split !='no_test':
            test_loader = DataLoader(self.dataset_processed['test'],
                            batch_size=batch_size, shuffle=False)
            self.dataloader =  {'train_loader': train_loader,
                                'val_loader': val_loader,
                                'test_loader': test_loader}

        else: 
            self.dataloader =  {'train_loader': train_loader,
                                'val_loader': val_loader}
        print_sys(f"Loader sizes - Train: {len(train_loader.dataset)}, Validation: {len(val_loader.dataset)}, "
            f"Test: {len(test_loader.dataset) if self.split != 'no_test' else 'N/A'}")
        print_sys("Done!")
            
    def create_cell_graph(self, X, y, de_idx, pert, pert_idx,
            dosage, cov, embed_cell, mask):
        """
        Create a cell graph from a given cell

        Parameters
        ----------
        X: np.ndarray
            Gene expression matrix
        y: np.ndarray
            Label vector
        de_idx: np.ndarray
            DE gene indices
        pert: str
            Perturbation category
        pert_idx: list
            List of perturbation indices

        Returns
        -------
        torch_geometric.data.Data
            Cell graph to be used in dataloader

        """

        feature_mat = torch.Tensor(X.reshape(1,-1))
        y = torch.Tensor(y.reshape(1,-1))
        mask = torch.Tensor(mask.reshape(1,-1)).bool()
        embed_cell = torch.Tensor(embed_cell.reshape(1,-1))

        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=y, de_idx=de_idx, pert=pert, 
                    dosage=torch.tensor(dosage), cov=torch.tensor(cov), 
                    embed_cell=embed_cell, mask=mask)

    def create_cell_graph_dataset(self, split_adata, perturbation,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs

        Parameters
        ----------
        split_adata: anndata.AnnData
            Annotated data matrix
        perturbation: str
            Perturbation category
        num_samples: int
            Number of samples to create per perturbed cell (i.e. number of
            control cells to map to each perturbed cell)

        Returns
        -------
        list
            List of cell graphs

        """

        num_de_genes = 20        
        adata_ = split_adata[split_adata.obs['perturbation'] == perturbation]
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []
        dosages = []
        covs = []
        embed_cells = []
        masks = []

        # When considering a non-control perturbation
        if perturbation != 'control':
            # Get the indices of applied perturbation
            pert_idx = [np.where(p == self.pert_names)[0][0] for p in perturbation.split('+') if p in self.pert_names]

            # Store list of genes that are most differentially expressed for testing            
            if de:
                pert_de_category = adata_.obs['condition_name'][0]
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
            
            for dataset in adata_.obs['dataset'].unique():
                ctrl_adata = self.ctrl_adata[self.ctrl_adata.obs['dataset']==dataset]
                for i, cell_z in enumerate(adata_[adata_.obs['dataset']==dataset].X):
                    # Use samples from control as basal expression
                    ctrl_samples = ctrl_adata[np.random.randint(0, len(ctrl_adata), num_samples), :]
                    for c in ctrl_samples.X:
                        Xs.append(c)
                        ys.append(cell_z)
                        dosages.append(adata_.obs.iloc[i]['dosage'])
                        covs.append(adata_.obs.iloc[i][['id_dataset', 'id_celltype']])
                        embed_cells.append(adata_.uns['embed_cell'].loc[dataset].values)                        
                        masks.append(adata_.uns['mask'].loc[dataset].values)

        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for dataset in adata_.obs['dataset'].unique():
                for i, cell_z in enumerate(adata_[adata_.obs['dataset']==dataset].X):
                    Xs.append(cell_z)
                    ys.append(cell_z)
                    dosages.append(adata_.obs.iloc[i]['dosage'])
                    covs.append(adata_.obs.iloc[i][['id_dataset', 'id_celltype']])
                    embed_cells.append(adata_.uns['embed_cell'].loc[dataset].values)
                    masks.append(adata_.uns['mask'].loc[dataset].values)

        # Create cell graphs
        cell_graphs = [
            self.create_cell_graph(
            X.toarray(), y.toarray(), 
            de_idx, perturbation, pert_idx,
            dosage, cov, embed_cell, mask
            )
            for X, y, dosage, cov, embed_cell, mask in zip(Xs, ys, dosages, covs, embed_cells, masks)
        ]

        return cell_graphs

    def create_dataset_file(self):
        """
        Create dataset file for each perturbation condition
        """
        print_sys("Creating dataset file...")
        self.dataset_processed_all = {}
        for dataset in tqdm(self.set2conditions.keys(), desc="Datasets"):
            self.dataset_processed_all[dataset] = {}
            for p in tqdm(self.adata[self.adata.obs['dataset'] == dataset].obs['perturbation'].unique(), desc=f"Perturbations in {dataset}", leave=False):                
                self.dataset_processed_all[dataset][p] = self.create_cell_graph_dataset(self.adata[self.adata.obs['dataset']==dataset], p)
        print_sys("Done!")

    def split_dataset_file(self):
        print_sys("Splitting dataset file...")
        self.dataset_processed = {}
        for split in tqdm(self.adata.obs['split'].unique(), desc="Splits"):
            self.dataset_processed[split] = [self.dataset_processed_all[dataset][p] for dataset in self.adata.obs['dataset'].unique() for p in self.set2conditions[dataset][split]]
            self.dataset_processed[split] = sum(self.dataset_processed[split], [])
        print_sys("Done!")