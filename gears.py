from copy import deepcopy
import os
import pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader

from .model import GEARS_Model, GNN_Model
from .inference import evaluate, compute_metrics
from .utils import loss_fct, uncertainty_loss_fct, \
                  get_similarity_network, print_sys, GeneSimNetwork, \
                  create_cell_graph_dataset_for_prediction, get_mean_control, \
                  get_GI_genes_idx, get_GI_params, get_per_gene_thresholds

torch.manual_seed(0)

import warnings
warnings.filterwarnings("ignore")

class GEARS:
    """
    GEARS base model class
    """

    def __init__(self, pert_data,
                 G_coexpress_edge_list = None,
                 device = 'cuda',
                 weight_bias_track = False, 
                 proj_name = 'GEARS', 
                 exp_name = 'GEARS'):
        """
        Initialize GEARS model

        Parameters
        ----------
        pert_data: PertData object
            dataloader for perturbation data
        device: str
            Device to run the model on. Default: 'cuda'
        weight_bias_track: bool
            Whether to track performance on wandb. Default: False
        proj_name: str
            Project name for wandb. Default: 'GEARS'
        exp_name: str
            Experiment name for wandb. Default: 'GEARS'

        Returns
        -------
        None

        """

        self.weight_bias_track = weight_bias_track
        
        if self.weight_bias_track:
            import wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None
        
        self.device = device
        self.config = None
        
        self.dataloader = pert_data.dataloader
        self.adata = pert_data.adata
        self.node_map = pert_data.node_map
        self.node_map_pert = pert_data.node_map_pert
        self.data_path = pert_data.data_path
        self.dataset_name = pert_data.dataset_name
        self.split = pert_data.split
        self.seed = pert_data.seed
        self.train_gene_set_size = pert_data.train_gene_set_size
        self.set2conditions = pert_data.set2conditions
        self.subgroup = pert_data.subgroup
        self.gene_list = pert_data.gene_names.values.tolist()
        self.pert_list = pert_data.pert_names.tolist()
        self.num_genes = len(self.gene_list)
        self.num_perts = len(self.pert_list)
        self.default_pert_graph = pert_data.default_pert_graph
        self.saved_pred = {}
        self.saved_logvar_sum = {}
        
        self.ctrl_expression = torch.tensor(
            np.mean(self.adata.X[self.adata.obs['perturbation'].values == 'control'],
                    axis=0)).reshape(-1, ).to(self.device)
        self.ctrl_adata = self.adata[self.adata.obs['perturbation'].values == 'control']

        gene_dict = {g:i for i,g in enumerate(self.gene_list)}
        self.pert2gene = {p: gene_dict[pert] for p, pert in
                          enumerate(self.pert_list) if pert in self.gene_list}
        
        self.G_coexpress_edge_list = G_coexpress_edge_list


    
    def tunable_parameters(self):
        """
        Return the tunable parameters of the model

        Returns
        -------
        dict
            Tunable parameters of the model

        """

        return {'hidden_size': 'hidden dimension, default 64',
                'num_go_gnn_layers': 'number of GNN layers for GO graph, default 1',
                'num_gene_gnn_layers': 'number of GNN layers for co-expression gene graph, default 1',
                'decoder_hidden_size': 'hidden dimension for gene-specific decoder, default 16',
                'num_similar_genes_go_graph': 'number of maximum similar K genes in the GO graph, default 20',
                'num_similar_genes_co_express_graph': 'number of maximum similar K genes in the co expression graph, default 20',
                'coexpress_threshold': 'pearson correlation threshold when constructing coexpression graph, default 0.4',
                'uncertainty': 'whether or not to turn on uncertainty mode, default False',
                'uncertainty_reg': 'regularization term to balance uncertainty loss and prediction loss, default 1',
                'direction_lambda': 'regularization term to balance direction loss and prediction loss, default 1'
               }
    
    def model_initialize(self, hidden_size = 64,
                         num_go_gnn_layers = 1, 
                         num_gene_gnn_layers = 1,
                         decoder_hidden_size = 16,
                         num_similar_genes_go_graph = 20,
                         num_similar_genes_co_express_graph = 20,                    
                         coexpress_threshold = 0.4,
                         uncertainty = False, 
                         uncertainty_reg = 1,
                         direction_lambda = 1e-1,
                         G_go = None,
                         G_go_weight = None,
                         G_coexpress_edge_list = None,
                         G_coexpress = None,
                         G_coexpress_weight = None,
                         no_perturb = False,
                         model = 'GEARS',
                         **kwargs
                        ):
        """
        Initialize the model

        Parameters
        ----------
        hidden_size: int
            hidden dimension, default 64
        num_go_gnn_layers: int
            number of GNN layers for GO graph, default 1
        num_gene_gnn_layers: int
            number of GNN layers for co-expression gene graph, default 1
        decoder_hidden_size: int
            hidden dimension for gene-specific decoder, default 16
        num_similar_genes_go_graph: int
            number of maximum similar K genes in the GO graph, default 20
        num_similar_genes_co_express_graph: int
            number of maximum similar K genes in the co expression graph, default 20
        coexpress_threshold: float
            pearson correlation threshold when constructing coexpression graph, default 0.4
        uncertainty: bool
            whether or not to turn on uncertainty mode, default False
        uncertainty_reg: float
            regularization term to balance uncertainty loss and prediction loss, default 1
        direction_lambda: float
            regularization term to balance direction loss and prediction loss, default 1
        G_go: scipy.sparse.csr_matrix
            GO graph, default None
        G_go_weight: scipy.sparse.csr_matrix
            GO graph edge weights, default None
        G_coexpress: scipy.sparse.csr_matrix
            co-expression graph, default None
        G_coexpress_weight: scipy.sparse.csr_matrix
            co-expression graph edge weights, default None
        no_perturb: bool
            predict no perturbation condition, default False

        Returns
        -------
        None
        """
        
        self.config = {'hidden_size': hidden_size,
                       'num_go_gnn_layers' : num_go_gnn_layers, 
                       'num_gene_gnn_layers' : num_gene_gnn_layers,
                       'decoder_hidden_size' : decoder_hidden_size,
                       'num_similar_genes_go_graph' : num_similar_genes_go_graph,
                       'num_similar_genes_co_express_graph' : num_similar_genes_co_express_graph,
                       'coexpress_threshold': coexpress_threshold,
                       'uncertainty' : uncertainty, 
                       'uncertainty_reg' : uncertainty_reg,
                       'direction_lambda' : direction_lambda,
                       'G_go': G_go,
                       'G_go_weight': G_go_weight,
                       'G_coexpress': G_coexpress,
                       'G_coexpress_weight': G_coexpress_weight,
                       'device': self.device,
                       'num_genes': self.num_genes,
                       'num_perts': self.num_perts,
                       'no_perturb': no_perturb,
                       'model': model
                      }
        
        if self.wandb:
            self.wandb.config.update(self.config)
        
        if self.config['G_coexpress'] is None:

            if self.G_coexpress_edge_list is None:
            ## calculating co expression similarity graph
                edge_list = get_similarity_network(network_type='co-express',
                                                adata=self.adata,
                                                threshold=coexpress_threshold,
                                                k=num_similar_genes_co_express_graph,
                                                data_path=self.data_path,
                                                data_name=self.dataset_name,
                                                split=self.split, seed=self.seed,
                                                train_gene_set_size=self.train_gene_set_size,
                                                set2conditions=self.set2conditions)
            else:
                edge_list = self.G_coexpress_edge_list

            sim_network = GeneSimNetwork(edge_list, self.gene_list, node_map = self.node_map)
            self.config['G_coexpress'] = sim_network.edge_index
            self.config['G_coexpress_weight'] = sim_network.edge_weight
        
        if self.config['G_go'] is None:
            ## calculating gene ontology similarity graph
            edge_list = get_similarity_network(network_type='go',
                                               adata=self.adata,
                                               threshold=coexpress_threshold,
                                               k=num_similar_genes_go_graph,
                                               pert_list=self.pert_list,
                                               data_path=self.data_path,
                                               data_name=self.dataset_name,
                                               split=self.split, seed=self.seed,
                                               train_gene_set_size=self.train_gene_set_size,
                                               set2conditions=self.set2conditions,
                                               default_pert_graph=self.default_pert_graph)

            sim_network = GeneSimNetwork(edge_list, self.pert_list, node_map = self.node_map_pert)
            self.config['G_go'] = sim_network.edge_index
            self.config['G_go_weight'] = sim_network.edge_weight
            
        if self.config['model'] == 'GEARS':
            self.model = GEARS_Model(self.config).to(self.device)
        elif self.config['model'] == 'GNN':
            self.model = GNN_Model(self.config).to(self.device)

            thresholds = get_per_gene_thresholds(self.dataloader['train_loader'], self.num_genes)            
            self.model.load_cov_encoder(num_batches=10, num_celllines=3)
            self.model.load_threshold_encoder(thresholds=thresholds)

        self.best_model = deepcopy(self.model)

        
    def load_pretrained(self, path):
        """
        Load pretrained model

        Parameters
        ----------
        path: str
            path to the pretrained model

        Returns
        -------
        None
        """

        with open(os.path.join(path, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        del config['device'], config['num_genes'], config['num_perts']
        self.model_initialize(**config)
        
        state_dict = torch.load(os.path.join(path, 'model.pt'), map_location = torch.device('cpu'))
        if next(iter(state_dict))[:7] == 'module.':
            # the pretrained model is from data-parallel module
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.best_model = self.model
    
    def save_model(self, path):
        """
        Save the model

        Parameters
        ----------
        path: str
            path to save the model

        Returns
        -------
        None

        """
        if not os.path.exists(path):
            os.mkdir(path)
        
        if self.config is None:
            raise ValueError('No model is initialized...')
        
        with open(os.path.join(path, 'config.pkl'), 'wb') as f:
            pickle.dump(self.config, f)
       
        torch.save(self.best_model.state_dict(), os.path.join(path, 'model.pt'))
    
    def predict(self, pert_list):
        """
        Predict the transcriptome given a list of genes/gene combinations being
        perturbed

        Parameters
        ----------
        pert_list: list
            list of genes/gene combiantions to be perturbed

        Returns
        -------
        results_pred: dict
            dictionary of predicted transcriptome
        results_logvar: dict
            dictionary of uncertainty score

        """
        ## given a list of single/combo genes, return the transcriptome
        ## if uncertainty mode is on, also return uncertainty score.
        for pert in pert_list:
            for i in pert:
                if i not in self.pert_list:
                    raise ValueError(i+ " is not in the perturbation graph. "
                                        "Please select from GEARS.pert_list!")
        
        if self.config['uncertainty']:
            results_logvar = {}
            
        self.best_model = self.best_model.to(self.device)
        self.best_model.eval()
        results_pred = {}
        results_logvar_sum = {}
        
        for pert in pert_list:
            try:
                #If prediction is already saved, then skip inference
                results_pred['_'.join(pert)] = self.saved_pred['_'.join(pert)]
                if self.config['uncertainty']:
                    results_logvar_sum['_'.join(pert)] = self.saved_logvar_sum['_'.join(pert)]
                continue
            except:
                pass
            
            cg = create_cell_graph_dataset_for_prediction(pert, self.ctrl_adata,
                                                    self.pert_list, self.device)
            loader = DataLoader(cg, 300, shuffle = False)
            batch = next(iter(loader))
            batch.to(self.device)

            with torch.no_grad():
                if self.config['uncertainty']:
                    p, unc = self.best_model(batch)
                    results_logvar['_'.join(pert)] = np.mean(unc.detach().cpu().numpy(), axis = 0)
                    results_logvar_sum['_'.join(pert)] = np.exp(-np.mean(results_logvar['_'.join(pert)]))
                else:
                    p = self.best_model(batch)
                    
            results_pred['_'.join(pert)] = np.mean(p.detach().cpu().numpy(), axis = 0)
                
        self.saved_pred.update(results_pred)
        
        if self.config['uncertainty']:
            self.saved_logvar_sum.update(results_logvar_sum)
            return results_pred, results_logvar_sum
        else:
            return results_pred
        
    def GI_predict(self, combo, GI_genes_file='./genes_with_hi_mean.npy'):
        """
        Predict the GI scores following perturbation of a given gene combination

        Parameters
        ----------
        combo: list
            list of genes to be perturbed
        GI_genes_file: str
            path to the file containing genes with high mean expression

        Returns
        -------
        GI scores for the given combinatorial perturbation based on GEARS
        predictions

        """

        ## if uncertainty mode is on, also return uncertainty score.
        try:
            # If prediction is already saved, then skip inference
            pred = {}
            pred[combo[0]] = self.saved_pred[combo[0]]
            pred[combo[1]] = self.saved_pred[combo[1]]
            pred['_'.join(combo)] = self.saved_pred['_'.join(combo)]
        except:
            if self.config['uncertainty']:
                pred = self.predict([[combo[0]], [combo[1]], combo])[0]
            else:
                pred = self.predict([[combo[0]], [combo[1]], combo])

        mean_control = get_mean_control(self.adata).values  
        pred = {p:pred[p]-mean_control for p in pred} 

        if GI_genes_file is not None:
            # If focussing on a specific subset of genes for calculating metrics
            GI_genes_idx = get_GI_genes_idx(self.adata, GI_genes_file)       
        else:
            GI_genes_idx = np.arange(len(self.adata.var.gene_name.values))
            
        pred = {p:pred[p][GI_genes_idx] for p in pred}
        return get_GI_params(pred, combo)
    
    
    def train(self, epochs = 20, 
              lr = 1e-3,
              weight_decay = 5e-4
             ):
        """
        Train the model

        Parameters
        ----------
        epochs: int
            number of epochs to train
        lr: float
            learning rate
        weight_decay: float
            weight decay

        Returns
        -------
        None

        """
        
        train_loader = self.dataloader['train_loader']
        val_loader = self.dataloader['val_loader']
            
        self.model = self.model.to(self.device)
        best_model = deepcopy(self.model)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        min_val = np.inf
        print_sys('Start Training...')

        for epoch in range(epochs):
            self.model.train()

            for step, batch in enumerate(train_loader):
                batch.to(self.device)
                optimizer.zero_grad()
                y = batch.y

                if 'mask' in batch.keys():
                    mask = batch.mask
                else: 
                    mask = None

                if self.config['uncertainty']:
                    pred, logvar = self.model(batch)
                    loss = uncertainty_loss_fct(pred, logvar, y, batch.pert,
                                      reg = self.config['uncertainty_reg'],
                                      ctrl = self.ctrl_expression, 
                                      mask = mask,
                                      direction_lambda = self.config['direction_lambda'])
                else:
                    pred = self.model(batch)
                    loss = loss_fct(pred, y, batch.pert,
                                  ctrl = self.ctrl_expression, 
                                  mask = mask,
                                  direction_lambda = self.config['direction_lambda'])
                loss.backward()
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()

                if self.wandb:
                    self.wandb.log({'training_loss': loss.item()})

                if step % 50 == 0:
                    log = "Epoch {} Step {} Train Loss: {:.4f}" 
                    print_sys(log.format(epoch + 1, step + 1, loss.item()))

            scheduler.step()
            # Evaluate model performance on train and val set
            train_res = evaluate(train_loader, self.model,
                                 self.config['uncertainty'], self.device)
            val_res = evaluate(val_loader, self.model,
                                 self.config['uncertainty'], self.device)
            train_metrics, _ = compute_metrics(train_res)
            val_metrics, _ = compute_metrics(val_res)

            # Print epoch performance
            log = "Epoch {}: Train Overall MSE: {:.4f} " \
                  "Validation Overall MSE: {:.4f}. "
            print_sys(log.format(epoch + 1, train_metrics['mse'], 
                             val_metrics['mse']))
            
            # # Print epoch performance for DE genes
            # log = "Train Top 20 DE MSE: {:.4f} " \
            #       "Validation Top 20 DE MSE: {:.4f}. "
            # print_sys(log.format(train_metrics['mse_de'],
            #                  val_metrics['mse_de']))
            
            if self.wandb:
                metrics = ['mse', 'pearson']
                for m in metrics:
                    self.wandb.log({'train_' + m: train_metrics[m],
                               'val_'+m: val_metrics[m]})
                
        print_sys("Done!")
        self.best_model = best_model

        if 'test_loader' not in self.dataloader:
            print_sys('Done! No test dataloader detected.')
            return
            
        print_sys('Done!')
