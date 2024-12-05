import sys
# sys.path.append("../")
import json
import torch
import pandas as pd
from tqdm import tqdm
from HAMUR.basic.features import DenseFeature, SparseFeature
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from HAMUR.trainers import CTRTrainer
from HAMUR.utils.data import DataGenerator
from HAMUR.models.multi_domain import Mlp_7_Layer, Mlp_2_Layer, MLP_adap_2_layer_1_adp, DCN_MD, DCN_MD_adp, WideDeep_MD, WideDeep_MD_adp

def get_movielens_data_rank_multidomain(data_path):
    train_data = pd.read_csv(data_path+"/train_inter.csv")
    valid_data = pd.read_csv(data_path+"/valid_inter.csv")
    test_data = pd.read_csv(data_path+"/test_inter.csv")
    data = pd.concat([train_data, valid_data, test_data])
    domain_weight = train_data['domain_id'].value_counts(normalize=True).to_dict()

    domain_num = 4
    sparse_features = ['user_id', 'item_id', "domain_id"]

    dense_feas = []
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]


    target = 'label'
    train_y = train_data[target]
    del train_data[target]
    valid_y = valid_data[target]
    del valid_data[target]
    test_y = test_data[target]
    del test_data[target]


    return dense_feas, sparse_feas, train_data, valid_data, test_data, train_y, valid_y, test_y, domain_num, domain_weight

def map_group_indicator(age, list_group):
    l = len(list(list_group))
    for i in range(l):
        if age in list_group[i]:
            return i


def convert_target(val):
    v = int(val)
    if v > 3:
        return int(1)
    else:
        return int(0)


def convert_numeric(val):
    """
    Forced conversion
    """
    return int(val)


def df_to_dict_multi_domain(data, columns):
    """
    Convert the array to a dict type input that the network can accept
    Args:
        data (array): 3D datasets of type DataFrame (Length * Domain_num * feature_num)
        columns (list): feature name list
    Returns:
        The converted dict, which can be used directly into the input network
    """

    data_dict = {}
    for i in range(len(columns)):
        data_dict[columns[i]] = data[:, :, i]
    return data_dict

def main(dataset_path, model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir, seed):
    torch.manual_seed(seed)
    dense_feas, sparse_feas, train_x, valid_x, test_x, train_label, valid_label, test_label, domain_num, domain_weight = get_movielens_data_rank_multidomain(dataset_path)
    dg = DataGenerator(train_x, train_label)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(batch_size=batch_size, x_test=test_x, x_val=valid_x,
                                                                               y_test=test_label, y_val=valid_label)
    if model_name == "mlp":
        model = Mlp_2_Layer(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128])
    elif model_name == "mlp_adp":
        model = MLP_adap_2_layer_1_adp(dense_feas + sparse_feas, domain_num=domain_num, fcn_dims=[256, 128],
                                       hyper_dims=[64], k=35)
    elif model_name == "dcn_md":
        model = DCN_MD(features=dense_feas + sparse_feas,num_domains=domain_num ,n_cross_layers=2, mlp_params={"dims": [256, 128]})
    elif model_name == "dcn_md_adp":
        model = DCN_MD_adp(features=dense_feas + sparse_feas,num_domains=domain_num, n_cross_layers=2, k = 30, mlp_params={"dims": [256, 128]}, hyper_dims=[128])
    elif model_name == "wd_md":
        model = WideDeep_MD(wide_features=dense_feas,num_domains= domain_num, deep_features=sparse_feas, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif model_name == "wd_md_adp":
        model = WideDeep_MD_adp(wide_features=dense_feas,num_domains= domain_num, deep_features=sparse_feas,  k= 45,mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}, hyper_dims=[128])
    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=10, device=device, model_path=save_dir,scheduler_params={"step_size": 4,"gamma": 0.85}, domain_weight=domain_weight)
    #scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    ctr_trainer.fit(train_dataloader, val_dataloader)
    metric_dict = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
    print(metric_dict)
    # auc1,auc2,auc3,auc4,auc = ctr_trainer.evaluate_multi_domain_auc(ctr_trainer.model, test_dataloader)
    # log1,log2,log3,log4,log = ctr_trainer.evaluate_multi_domain_logloss(ctr_trainer.model, test_dataloader)
    # print(f'test auc: {auc} | test logloss: {log}')
    # print(f'domain 1 test auc: {auc1} | test logloss: {log1}')
    # print(f'domain 2 test auc: {auc2} | test logloss: {log2}')
    # print(f'domain 3 test auc: {auc3} | test logloss: {log3}')
    # print(f'domain 4 test auc: {auc4} | test logloss: {log4}')

    # save csv file
    # import csv
    # with open(model_name+"_"+str(seed)+'.csv', "w", newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['model', 'seed', 'auc', 'log', 'auc1', 'log1', 'auc2', 'log2', 'auc3', 'log3'])
    #     writer.writerow([model_name, str(seed), auc, log, auc1,log1,auc2,log2,auc3,log3])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="./data")
    parser.add_argument('--model_name', default='mlp_adp')
    parser.add_argument('--epoch', type=int, default=100)  #100
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4096)  #4096
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', default='cpu')  #cuda:0
    parser.add_argument('--save_dir', default='./')
    parser.add_argument('--seed', type=int, default=2024)

    args = parser.parse_args()
    main(args.dataset_path, args.model_name, args.epoch, args.learning_rate, args.batch_size, args.weight_decay, args.device, args.save_dir, args.seed)
