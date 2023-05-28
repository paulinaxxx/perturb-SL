import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as du
from NN import *

def data_shuffle(data):
    np.random.seed(527)
    indices = np.random.permutation(len(data))
    data = data[indices]
    return data

def pert_split(pert_data, split_method):
    np.random.seed(257)
    pert_arr = pert_data[:,1:]
    pert_list, indices = np.unique(pert_arr, axis=0, return_inverse=True)
    pert_df = pd.DataFrame({"viab":pert_data[:,0], "pert_1":pert_arr[:,0], "pert_2":pert_arr[:,1], "idx":indices})
    unique_idx = [i for i in range(len(pert_list))]
    np.random.shuffle(unique_idx)

    i = split_method*len(pert_list)
    pert_test = unique_idx[i:i+int(len(pert_list)/5)]
    pert_df_test = pert_df[[True if i in pert_test else False for i in pert_df["idx"]]]
    pert_train = unique_idx[i+int(len(pert_list)/5):]
    pert_df_train = pert_df[[True if i in pert_train else False for i in pert_df["idx"]]]
    
    return pert_df_test.to_numpy()[:,:-1], pert_df_train.to_numpy()[:,:-1]

def pert_onehot(pert_data, gene_dim):
    pert_onehot_data = np.zeros((len(pert_data), gene_dim))
    for i, line in enumerate(pert_data):
        pert_onehot_data[i][int(line[0])] = 1
        if int(line[1]) != -1:
            pert_onehot_data[i][int(line[1])] = 1
    return pert_onehot_data

def prepare_data(ctrl_file, single_pert_file, double_pert_file, split_method=0):
    ctrl_data = np.load(ctrl_file)
    single_pert_data = np.load(single_pert_file)
    double_pert_data = np.load(double_pert_file)

    single_pert_test, single_pert_train = pert_split(single_pert_data, split_method)
    double_pert_test, double_pert_train = pert_split(double_pert_data, split_method)

    for data in [single_pert_test, single_pert_train, double_pert_test, double_pert_train]:
        data = data_shuffle(data)
    pert_train = np.concatenate((single_pert_train, double_pert_train), axis=0)
    pert_test = np.concatenate((single_pert_test, double_pert_test), axis=0)
    
    np.random.seed(725)
    train_size = len(pert_train)
    test_size = len(pert_test)
    total_size = train_size+test_size
    i = split_method*len(ctrl_data)
    ctrl_random_idx = np.random.choice(len(ctrl_data), 5*len(ctrl_data))
    ctrl_data_test = np.zeros((test_size, ctrl_data.shape[1]))
    for count,index in enumerate(ctrl_random_idx[i:i+test_size]):
        ctrl_data_test[count] = ctrl_data[index]
    ctrl_data_train = np.zeros((train_size, ctrl_data.shape[1]))
    for count,index in enumerate(ctrl_random_idx[i+test_size:i+total_size]):
        ctrl_data_train[count] = ctrl_data[index]

    ctrl_data_test = ctrl_data_test[:,3:]
    ctrl_data_train = ctrl_data_train [:,3:]

    onehot_test = pert_onehot(pert_test[:,1:], ctrl_data_test.shape[1])
    onehot_train = pert_onehot(pert_train[:,1:], ctrl_data_train.shape[1])

    train_feature = np.concatenate((ctrl_data_train, onehot_train), axis=1)
    test_feature = np.concatenate((ctrl_data_test, onehot_test), axis=1)
    train_label = pert_train[:,0]
    test_label = pert_test[:,0]

    return torch.Tensor(train_feature), torch.Tensor(train_label).reshape((train_feature.shape[0],1)), torch.Tensor(test_feature), torch.Tensor(test_label).reshape((test_feature.shape[0],1))


def construct_input_mask(term_gene_dict, ngenes, gene2id_dict):
    term_mask_map = {}
    for term, gene_set in term_gene_dict.items():
        mask = torch.zeros(len(gene_set), ngenes)
        gene_set_id = [gene2id_dict[g] for g in gene_set]
        for i, gene_id in enumerate(gene_set_id):
            mask[i, gene_id] = 1

        mask_gpu = torch.autograd.Variable(mask.cuda())
        term_mask_map[term] = mask_gpu

    return term_mask_map

def train_model(dG, input_gene, gene2id_dict, train_data):
    # set training parameters
    learning_rate = 0.001
    batch_size = 5000
    train_epochs = 300
    max_corr = 0

    best_model = 0
    model = nn_model(dG, input_gene, gene2id_dict)

    train_feature, train_label, test_feature, test_label = train_data

    train_label_gpu = torch.autograd.Variable(train_label.cuda())
    test_label_gpu = torch.autograd.Variable(test_label.cuda())

    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-05)
    term_mask_map = construct_input_mask(model.term_gene_dict, len(input_gene), gene2id_dict)

    optimizer.zero_grad()

    for name, param in model.named_parameters():
        term_name = name.split('_')[0]

        if '_gene_layer.weight' in name:
            param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
        else:
            param.data = param.data * 0.1

    train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
    test_loader = du.DataLoader(du.TensorDataset(test_feature,test_label), batch_size=batch_size, shuffle=False)

    for epoch in range(train_epochs):

        model.train()
        train_predict = torch.zeros(0,0).cuda()

        for i, (inputdata, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            features = inputdata

            cuda_features = torch.autograd.Variable(features.cuda())
            cuda_labels = torch.autograd.Variable(labels.cuda())

            optimizer.zero_grad()  # zero the gradient buffer

            aux_ouput_dict, _, _ = model(cuda_features)

            if train_predict.size()[0] == 0:
                train_predict = aux_ouput_dict['final'].data
            else:
                train_predict = torch.cat([train_predict, aux_ouput_dict['final'].data], dim=0)

            total_loss = 0	
            for name, output in aux_ouput_dict.items():
                loss = nn.MSELoss()
                if name == 'final':
                    total_loss += loss(output, cuda_labels)
                else: # change 0.2 to smaller one for big terms
                    total_loss += 0.2 * loss(output, cuda_labels)

            total_loss.backward()

            for name, param in model.named_parameters():
                if '_gene_layer.weight' not in name:
                    continue
                term_name = name.split('_')[0]
                #print name, param.grad.data.size(), term_mask_map[term_name].size()
                param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

            optimizer.step()

        train_corr = pearson_corr(train_predict, train_label_gpu)

        #Test: random variables in training mode become static
        model.eval()
        
        test_predict = torch.zeros(0,0).cuda()

        for i, (inputdata, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            features = inputdata
            cuda_features = Variable(features.cuda())

            aux_out_map, _, _ = model(cuda_features)

            if test_predict.size()[0] == 0:
                test_predict = aux_out_map['final'].data
            else:
                test_predict = torch.cat([test_predict, aux_out_map['final'].data], dim=0)

        test_corr = pearson_corr(test_predict, test_label_gpu)

        print("epoch\t%d\ttrain_corr\t%.6f\tval_corr\t%.6f\ttotal_loss\t%.6f" % (epoch, train_corr, test_corr, total_loss))

        if test_corr >= max_corr:
            max_corr = test_corr
            best_model = epoch

    # torch.save(model, 'result/model/model_final.pt')	

    print("Best performed model (epoch)\t%d" % best_model)


if __name__ == "__main__":
    GOA_df_filepath = "./result/goa_human_filtered.csv"
    OBO_filepath = "data/GO/go.obo"
    genelist_filepath = "result/gene2id.csv"

    dG, glist = build_go_graph(GOA_df_filepath, OBO_filepath, genelist_filepath)
    
    gene2id_df = pd.read_csv(genelist_filepath)
    gene2id_dict = dict(zip(list(gene2id_df["gene"]), list(gene2id_df["id"])))

    ctrl_file = "result/input/norm_input_ctrl.npy"
    single_pert_file = "result/input/norm_label_single.npy"
    double_pert_file = "result/input/norm_label_double.npy"
    train_data = prepare_data(ctrl_file, single_pert_file, double_pert_file, split_method=0)

    train_model(dG, glist, gene2id_dict, train_data)