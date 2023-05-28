import pandas as pd
import networkx as nx
from goatools import obo_parser
import torch
import torch.nn as nn
import torch.nn.functional as F

def pearson_corr(x, y):
	xx = x - torch.mean(x)
	yy = y - torch.mean(y)

	return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))

def build_go_graph(GOA_filepath, OBO_filepath, genelist_filepath):
    GOA_df_filtered = pd.read_csv(GOA_filepath)
    OBO_file = obo_parser.GODag(OBO_filepath)
    GO_list = list(set(GOA_df_filtered["GO ID"]))
    glist_df = pd.read_csv(genelist_filepath) #gene2id
    glist = list(glist_df["gene"])

    dG = nx.DiGraph()
    for index,row in GOA_df_filtered.iterrows():
        dG.add_edge(row["DB Object Symbol"], row["GO ID"])
    for t in GO_list:
        term = OBO_file.query_term(t)
        for c in term.get_all_children():
                dG.add_edge(c, t)

    leaves_nodes = [n for n in dG.nodes if dG.in_degree(n) == 0]
    leaves_GO = [n for n in leaves_nodes if n not in glist]
    dG.remove_nodes_from(leaves_GO)

    sparse_nodes = ['GO:0007165', 'GO:0005515', 'GO:0010468', 'GO:1990837', 'GO:0009966', 'GO:0051179', 'GO:0043565', 'GO:0019222', 'GO:0003677', 'GO:0008150', 'GO:0003676', 'GO:0003674']
    dG.remove_nodes_from(sparse_nodes)
    
    return dG,glist


class nn_model(nn.Module):
    def __init__(self, dG, input_gene, gene2id_dict):
        super(nn_model, self).__init__()

        self.dG = dG
        self.gene_list = input_gene
        self.gene2id_map = gene2id_dict
        self.term_hiddens = 6

        self.construct_gene_pert_fuse_layer()
        self.contruct_gene_term_layer()
        self.construct_graph_nn()
         
        self.add_module('top_linear_layer', nn.Linear(self.top_input_len, 6))
        self.add_module('top_batchnorm_layer', nn.BatchNorm1d(6))
        self.add_module('top_aux_linear_layer', nn.Linear(6,1))
        self.add_module('top_linear_layer_output', nn.Linear(1, 1))

    def construct_gene_pert_fuse_layer(self):
        self.add_module('pert_linear_layer', nn.Linear(len(self.gene_list), len(self.gene_list)))

        # MLP
        self.add_module('pert_gene_linear0', nn.Linear(len(self.gene_list), 128))
        self.add_module('pert_gene_batch0', nn.BatchNorm1d(128))
        self.add_module('pert_gene_relu0', nn.ReLU())
        self.add_module('pert_gene_linear1', nn.Linear(128, 256))
        self.add_module('pert_gene_batch1', nn.BatchNorm1d(256))
        self.add_module('pert_gene_relu1', nn.ReLU())
        self.add_module('pert_gene_linear2', nn.Linear(256, 128))
        self.add_module('pert_gene_batch2', nn.BatchNorm1d(128))
        self.add_module('pert_gene_relu2', nn.ReLU())
        self.add_module('pert_gene_linear_out', nn.Linear(128, len(self.gene_list)))

    def contruct_gene_term_layer(self):
        g_list = self.gene_list
        self.term_gene_dict = {}
        for g in g_list:
            for t in self.dG.neighbors(g):
                if self.term_gene_dict.get(t):
                    self.term_gene_dict[t].append(g)
                else:
                    self.term_gene_dict[t] = []
                    self.term_gene_dict[t].append(g)

        for t in self.term_gene_dict.keys():
            self.add_module(t+'_gene_layer', nn.Linear(len(g_list), len(self.term_gene_dict[t])))
        
        self.dG.remove_nodes_from(g_list)

    def construct_graph_nn(self):
        self.term_layer_list = []
        self.term_neighbor_dict = {}
        for t in self.dG.nodes:
            self.term_neighbor_dict[t] = []
            for child in self.dG.predecessors(t):
                self.term_neighbor_dict[t].append(child)

        leaves_prev = [n for n in self.dG.nodes if self.dG.in_degree(n) == 0]
        leaves = leaves_prev

        layer_level = 1

        while True:
            if len(leaves)==0:
                self.top_input_len = len(leaves_prev)*self.term_hiddens
                self.term_layer_list = self.term_layer_list[:-1]
                break
            else:
                leaves_prev = leaves
                leaves = [n for n in self.dG.nodes if self.dG.in_degree(n) == 0]
                self.term_layer_list.append(leaves)

            for t in leaves:
                input_len = 0
                input_len += len(self.term_neighbor_dict[t])*self.term_hiddens
                
                if t in self.term_gene_dict.keys():
                    input_len += len(self.term_gene_dict[t])

                self.add_module(t+'_linear_layer', nn.Linear(input_len, self.term_hiddens))
                self.add_module(t+'_batchnorm_layer', nn.BatchNorm1d(self.term_hiddens))
                self.add_module(t+'_aux_linear_layer1', nn.Linear(self.term_hiddens,1))
                self.add_module(t+'_aux_linear_layer2', nn.Linear(1,1))

            self.dG.remove_nodes_from(leaves)
            layer_level += 1

    def forward(self, x):
        gene_input = x.narrow(1, 0, len(self.gene_list))
        pert_input = x.narrow(1, len(self.gene_list), len(self.gene_list))

        pert_linear_out = self._modules['pert_linear_layer'](pert_input)
        gene_pert_fuse = gene_input+pert_linear_out
        gene_pert_out0 = gene_pert_fuse
        for i in range(3):
            gene_pert_out1 = self._modules['pert_gene_linear'+str(i)](gene_pert_out0)
            gene_pert_out2 = self._modules['pert_gene_batch'+str(i)](gene_pert_out1)
            gene_pert_out3 = self._modules['pert_gene_relu'+str(i)](gene_pert_out2)
            gene_pert_out0 = gene_pert_out3
        gene_pert_fuse_out = self._modules['pert_gene_linear_out'](gene_pert_out3)

        gene_output_dict = {}
        for term, glist in self.term_gene_dict.items():
            gene_output_dict[term] = self._modules[term + '_gene_layer'](gene_pert_fuse_out)
        
        nn_output_dict = {}
        top_output_dict = {}
        aux_ouput_dict = {}

        for i, layer in enumerate(self.term_layer_list):
            if i == 4:
                print(layer)
            for term in layer:
                child_input_list = []

                for child in self.term_neighbor_dict[term]:
                    child_input_list.append(nn_output_dict[child])

                if term in self.term_gene_dict.keys():
                    child_input_list.append(gene_output_dict[term])

                child_input = torch.cat(child_input_list,1)

                nn_out = self._modules[term+'_linear_layer'](child_input)				

                Tanh_out = torch.tanh(nn_out)
                if i+1 == len(self.term_layer_list):
                    top_output_dict[term] = self._modules[term+'_batchnorm_layer'](Tanh_out)
                    aux_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](top_output_dict[term]))
                    aux_ouput_dict[term] = self._modules[term+'_aux_linear_layer2'](aux_out)
                else:
                    nn_output_dict[term] = self._modules[term+'_batchnorm_layer'](Tanh_out)
                    aux_out = torch.tanh(self._modules[term+'_aux_linear_layer1'](nn_output_dict[term]))
                    aux_ouput_dict[term] = self._modules[term+'_aux_linear_layer2'](aux_out)


        final_input = torch.cat(list(top_output_dict.values()),1)

        out = self._modules['top_batchnorm_layer'](torch.tanh(self._modules['top_linear_layer'](final_input)))
        top_output_dict['final'] = out

        aux_layer_out = torch.tanh(self._modules['top_aux_linear_layer'](out))
        aux_ouput_dict['final'] = self._modules['top_linear_layer_output'](aux_layer_out)

        return aux_ouput_dict, nn_output_dict, top_output_dict
    


# if __name__ == "__main__":
#     GOA_df_filepath = "./result/goa_human_filtered.csv"
#     OBO_filepath = "data/GO/go.obo"
#     dG,genelist = build_go_graph(GOA_df_filepath, OBO_filepath)

#     model = nn_model(dG,genelist)
#     model_named_modules = [x for x in model.named_modules()]
#     print(model_named_modules)
