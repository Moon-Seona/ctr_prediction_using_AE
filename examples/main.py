import torch
import tqdm
import random
import numpy as np

from pathlib import Path

from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

import copy

import sys
import os

import pandas as pd

from sklearn.model_selection import train_test_split


sys.path.append(os.path.abspath(os.path.dirname('__file__')))

#from torchfm.dataset.avazu import AvazuDataset
#from torchfm.dataset.criteo import CriteoDataset
#from torchfm.dataset.movielens import MovieLens1MDataset, MovieLens20MDataset

from torchfm.dataset.tpmn import TPMNDataset 

#### New model
from torchfm.model.tpmn_fm import FactorizationMachineModel as TPMNModel # fm
from torchfm.model.tpmn_dfm import DeepFactorizationMachineModel as TPMNModel2 # dfm
from torchfm.model.auto_fm import FactorizationMachineModel as auto_fm # reconstruct
from torchfm.model.auto_dfm import DeepFactorizationMachineModel as auto_dfm 
from torchfm.model.lstm_fm import FactorizationMachineModel as lstm_fm # not reconstruct
from torchfm.model.lstm_dfm import DeepFactorizationMachineModel as lstm_dfm 

#from torchfm.model.afi import AutomaticFeatureInteractionModel
#from torchfm.model.afm import AttentionalFactorizationMachineModel
#from torchfm.model.dcn import DeepCrossNetworkModel
#from torchfm.model.dfm import DeepFactorizationMachineModel
#from torchfm.model.ffm import FieldAwareFactorizationMachineModel
#from torchfm.model.fm import FactorizationMachineModel
#from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel
#from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
#from torchfm.model.lr import LogisticRegressionModel
#from torchfm.model.ncf import NeuralCollaborativeFiltering
#from torchfm.model.nfm import NeuralFactorizationMachineModel
#from torchfm.model.pnn import ProductNeuralNetworkModel
#from torchfm.model.wd import WideAndDeepModel
#from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
#from torchfm.model.afn import AdaptiveFactorizationNetwork


'''
from torchfm.model.tpmn_lr import LogisticRegressionModel as tpmn_lr
from torchfm.model.tpmn_ffm import FieldAwareFactorizationMachineModel as tpmn_ffm
from torchfm.model.tpmn_fnn import FactorizationSupportedNeuralNetworkModel as tpmn_fnn
from torchfm.model.tpmn_wd import WideAndDeepModel as tpmn_wd
from torchfm.model.tpmn_afm import AttentionalFactorizationMachineModel as tpmn_afm
from torchfm.model.tpmn_nfm import NeuralFactorizationMachineModel as tpmn_nfm
from torchfm.model.tpmn_fnfm import FieldAwareNeuralFactorizationMachineModel as tpmn_fnfm
from torchfm.model.tpmn_pnn import ProductNeuralNetworkModel as tpmn_pnn
from torchfm.model.tpmn_dcn import DeepCrossNetworkModel as tpmn_dcn
from torchfm.model.tpmn_xdfm import ExtremeDeepFactorizationMachineModel as tpmn_xdfm
from torchfm.model.tpmn_afi import AutomaticFeatureInteractionModel as tpmn_afi
from torchfm.model.tpmn_afn import AdaptiveFactorizationNetwork as tpmn_afn
from torchfm.model.lstm_ffm import FieldAwareFactorizationMachineModel as lstm_ffm
from torchfm.model.lstm_fnn import FactorizationSupportedNeuralNetworkModel as lstm_fnn
from torchfm.model.lstm_wd import WideAndDeepModel as lstm_wd
from torchfm.model.lstm_afm import AttentionalFactorizationMachineModel as lstm_afm
from torchfm.model.lstm_nfm import NeuralFactorizationMachineModel as lstm_nfm
from torchfm.model.lstm_fnfm import FieldAwareNeuralFactorizationMachineModel as lstm_fnfm
from torchfm.model.lstm_pnn import ProductNeuralNetworkModel as lstm_pnn
from torchfm.model.lstm_dcn import DeepCrossNetworkModel as lstm_dcn
from torchfm.model.lstm_xdfm import ExtremeDeepFactorizationMachineModel as lstm_xdfm
from torchfm.model.lstm_afi import AutomaticFeatureInteractionModel as lstm_afi
from torchfm.model.lstm_afn import AdaptiveFactorizationNetwork as lstm_afn
'''
####

SEED = 2020

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(SEED)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def get_dataset(name, path, cache):
    if name == 'movielens1M':
        return MovieLens1MDataset(path)
    elif name == 'movielens20M':
        return MovieLens20MDataset(path)
    elif name == 'criteo':
        return CriteoDataset(path)
    elif name == 'avazu':
        return AvazuDataset(path)
    elif name == 'tpmn': # tpmn
        return TPMNDataset(path, cache)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    field_dims = dataset.field_dims
    #print(field_dims)
    embed_dim = 16
    
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=16) # embed_dim=16
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
             field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        print("Model:AFN")
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400,400,400), dropouts=(0, 0, 0))
    
    # new model
    # category
#    elif name == 'tpmn_fm_original':
#        return FactorizationMachineModel(field_dims, embed_dim=20) # 16
#    elif name == 'tpmn_dfm_original': 
#        return DeepFactorizationMachineModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2) 
    elif name == 'tpmn_fm': 
        #print(dataset.new_dict)
        return TPMNModel(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim)
    elif name == 'tpmn_dfm' : 
        print(len(dataset.new_dict))
        return TPMNModel2(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, mlp_dims=(8, 8), dropout=0.7)
    
    elif name == 'tpmn_lr' : 
        return tpmn_lr(field_dims)
    elif name == 'tpmn_ffm': 
        return tpmn_ffm(field_dims, len(dataset.new_dict), embed_dim=16)
    elif name == 'tpmn_fnn' : # 바꿀 파라미터 없음..
        return tpmn_fnn(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'tpmn_wd': # 바꿀train_dataset = 파라미터 없음..
        return tpmn_wd(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'tpmn_afm' : # attn_size 줄여야할듯
        return tpmn_afm(field_dims, len(dataset.new_dict), embed_dim=16, attn_size=16, dropouts=(0.7, 0.7))
    elif name == 'tpmn_nfm' : # dropouts[0] 수정 가능? mlp_dims를 64로?
        return tpmn_nfm(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8,), dropouts=(0.7, 0.7)) 
    elif name == 'tpmn_fnfm' :
        return tpmn_fnfm(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8,), dropouts=(0.7, 0.7))
    elif name == 'tpmn_ipnn':
        return tpmn_pnn(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8,), method='inner', dropout=0.7)
    elif name == 'tpmn_opnn':
        return tpmn_pnn(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8,), method='outer', dropout=0.7)
    elif name == 'tpmn_dcn': # num_layers
        return tpmn_dcn(field_dims, len(dataset.new_dict), embed_dim=16, num_layers=3, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'tpmn_xdfm': # cross_layer_sizes
        return tpmn_xdfm(field_dims, len(dataset.new_dict), embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'tpmn_afi': # atten_embed_dim, num_heads, num_layers
        return tpmn_afi(field_dims, len(dataset.new_dict), embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(8, 8), dropouts=(0.7, 0.7, 0.7))
    elif name == 'tpmn_afn': # LNN_dim
        return tpmn_afn(field_dims, len(dataset.new_dict), embed_dim=16, LNN_dim=1500, mlp_dims=(8,8,8), dropouts=(0.7, 0.7, 0.7))
        
    # lstm
    elif name == 'lstm_fm' :
        return lstm_fm(field_dims, len(dataset.new_dict)+1, embed_dim=16)
    elif name == 'lstm_dfm': 
        return lstm_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=16, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'auto_fm' :
        return auto_fm(field_dims, len(dataset.new_dict)+1, embed_dim=16)
    elif name == 'auto_dfm': 
        return auto_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=16, mlp_dims=(8, 8), dropout=0.7)
    
    elif name == 'lstm_ffm' : # 아직 모델 완성 전
        return lstm_ffm(field_dims, len(dataset.new_dict), embed_dim=16)
    elif name == 'lstm_fnn' :
        return lstm_fnn(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'lstm_wd' :
        return lstm_wd(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'lstm_afm' :
        return lstm_afm(field_dims, len(dataset.new_dict), embed_dim=16, attn_size=16, dropouts=(0.7, 0.7))
    elif name == 'lstm_fnfm' :
        return lstm_fnfm(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8,), dropouts=(0.7, 0.7))
    elif name == 'lstm_ipnn':
        return lstm_pnn(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8,), method='inner', dropout=0.7)
    elif name == 'lstm_opnn':
        return lstm_pnn(field_dims, len(dataset.new_dict), embed_dim=16, mlp_dims=(8,), method='outer', dropout=0.7)
    elif name == 'lstm_dcn': # num_layers
        return lstm_dcn(field_dims, len(dataset.new_dict), embed_dim=16, num_layers=3, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'lstm_xdfm': # cross_layer_sizes
        return lstm_xdfm(field_dims, len(dataset.new_dict), embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(8, 8), dropout=0.7)
    elif name == 'lstm_afi': # atten_embed_dim, num_heads, num_layers
        return lstm_afi(field_dims, len(dataset.new_dict), embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(8, 8), dropouts=(0.7, 0.7, 0.7))
    elif name == 'lstm_afn': # LNN_dim
        return lstm_afn(field_dims, len(dataset.new_dict), embed_dim=16, LNN_dim=1500, mlp_dims=(8,8,8), dropouts=(0.7, 0.7, 0.7))
    
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, data_loader, criterion, device, log_interval=100):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0

def train_tpmn(model, optimizer, data_loader, val_data_loader, criterion, criterion2, criterion3, device, model_name, log_interval=1000):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_mse_loss = 0

    targets, predicts = list(), list()

    #for param in model.autoencoder.parameters():
    #    param.requires_grad = False

    #for param in model.encoder_linear.parameters():
    #    param.requires_grad = False

    #for param in model.encoder_linear2.parameters():
    #    param.requires_grad = False

    #for param in model.encoder_linear3.parameters():
    #    param.requires_grad = False

    #for param in model.embedding.parameters():
    #    param.requires_grad = False

    #model.autoencoder2.requires_grad = False
    #model.encoder_linear.requires_grad = False
    #model.encoder_linear2.requires_grad = False
    #model.encoder_linear3.requires_grad = False


    for i, (fields, target, additional_info) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        model.train()

        fields, target, additional_info = fields.to(device), target.to(device), additional_info.to(device)
        if model_name.startswith('tpmn'):
            y = model(fields, additional_info)

            loss = criterion(y, target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                print('    - loss:', total_loss / log_interval)
                total_loss = 0    
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
        elif model_name.startswith('lstm') or model_name.startswith('auto') :
            y, pred, hidden, encode, embed = model(fields, additional_info) 
            
            loss = criterion(y, target.float())

            mse_loss = criterion3(encode, embed)

            vocab = pred.shape[-1]
            
            appbundle = additional_info[:,:149].reshape(-1)
            pred = pred.view(-1, vocab)
            
            #carrier = additional_info[:,149:199].reshape(-1)
            #pred2 = pred.view(-1, vocab)

            #make = additional_info[:,199:].reshape(-1)
            #pred3 = pred3.view(-1, vocab)
            recon_loss = criterion2(pred, appbundle) # criterion2(pred2, carrier) +
            
            model.zero_grad()
            
            (loss + recon_loss).backward()

            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_mse_loss += mse_loss.item()

            targets.extend(target.tolist())
            predicts.extend(y.tolist())

            if (i + 1) % log_interval == 0:
                total_val_loss = 0
                model.eval()
                #for j, (fields, target, additional_info) in enumerate(tqdm.tqdm(val_data_loader, smoothing=0, mininterval=1.0)):
                #    fields, target, additional_info = fields.to(device), target.to(device), additional_info.to(device)
                #    if model_name != 'tpmn_fm' and model_name != 'tpmn_dfm':
                #        y = model(fields)
                #    else:
                #        y, val_pred, val_hidden, encode, embed = model(fields, additional_info)

                #    val_loss = criterion(y, target.float())
                #    total_val_loss += val_loss.item()

                accuracy = (appbundle == torch.max(pred, dim=1)[1]).float().sum() / appbundle.size(0)
                #accuracy2 = (carrier == torch.max(pred2, dim=1)[1]).float().sum() / carrier.size(0)
                #accuracy3 = (make == torch.max(pred3, dim=1)[1]).float().sum() / make.size(0)
                print('    - loss:', total_loss / log_interval, total_recon_loss / log_interval, '    - accuracy: ', accuracy.item()) 
                # accuracy2.item(),accuracy3.item(), total_mse_loss / log_interval, ,  '  - val loss:', total_val_loss
                total_loss = 0
                total_recon_loss = 0
                total_mse_loss = 0
        else:
            y = model(fields)
            loss = criterion(y, target.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                print('    - loss:', total_loss / log_interval)
                total_loss = 0    
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            
    return roc_auc_score(targets, predicts)
   
    #print("train roc: ", roc_auc_score(targets, predicts)) 

def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def test_tpmn(model, data_loader, device, model_name):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target, additional_info in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target, additional_info = fields.to(device), target.to(device), additional_info.to(device)
            if model_name.startswith('tpmn'):
                y = model(fields, additional_info)
            elif model_name.startswith('lstm') or model_name.startswith('auto') :
                y, _, _, _, _ = model(fields, additional_info)
            else:
                y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
        
    fp_rate, tp_rate, thresholds = roc_curve(targets, predicts)
    print(classification_report(targets, predicts > thresholds[np.argmax(tp_rate - fp_rate)]))
    print(confusion_matrix(targets, predicts > thresholds[np.argmax(tp_rate - fp_rate)]))
    
    return roc_auc_score(targets, predicts)

def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):

    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path, '.tpmn_train') 
    
    train_length = int(len(dataset) * 0.7)
    #valid_length = int(len(dataset) * 0.05)
    #test_length = int(len(dataset) * 0.05)
    test_length2 = len(dataset) - train_length #- valid_length - test_length
    train_dataset, test_dataset2 = torch.utils.data.random_split(
        dataset, (train_length, test_length2))
    
    t = np.array(train_dataset) # total shape 14,805,664 x 3
    #t = np.load('dataset.npz', allow_pickle=True)
    #t = t['t']
    #print(t['t'])
    
    category_data = np.array([row[0] for row in t])
    click_data = np.array([row[1] for row in t])
    additional_data = np.array([row[2] for row in t])
    
    sorted_index = np.argsort(category_data[:,1]) # carrier 4, 약 2000개
    
    sorted_category_data = category_data[sorted_index]
    sorted_click_data = click_data[sorted_index]
    sorted_additional_data = additional_data[sorted_index]
    
    sorted_split1 = int(train_length * 0.2)#7500000 # train to valid
    
    index = np.arange(len(t)) # just for concat
    t2 = [(sorted_category_data[i], sorted_click_data[i], sorted_additional_data[i]) for i in index]
    
    #train_dataset = t2[-sorted_split1:]

    #t3 = t2[:-sorted_split1]
    #random.shuffle(t3)
    #length = int((len(t2) - len(train_dataset))/2)
    #valid_dataset = t3[:length]
    #test_dataset = t3[length:]
    
    train_dataset = t2[:sorted_split1] + t2[sorted_split1*2:]

    t3 = t2[sorted_split1:sorted_split1*2]
    random.shuffle(t3)
    length = int(sorted_split1/2)
    valid_dataset = t3[:length]
    test_dataset = t3[length:]
    
    print('dataset length')
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True) # batch 에서 shuffle 필요
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    model = get_model(model_name, dataset).to(device) # dataset
    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.CrossEntropyLoss()
    criterion3 = torch.nn.MSELoss(size_average=False)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_auc = 0.0
    best_epoch = -1

    #trained_model = torch.load(f'{save_dir}/{model_name}2.pt')

    #for key in trained_model.state_dict().keys():
    #    if key.startswith('autoencoder2') or key.startswith('encoder_linear'):
    #        model.state_dict()[key].copy_(trained_model.state_dict()[key])

    train = True
    count = 0
    
    train_auc_list = []
    auc_list = []
    #auc_list.append(auc)
    #for i in range(len(auc_list)) :
    #    print(i, 'epoch : ', auc_list[i])
    
    if train:
        for epoch_i in range(epoch):
            if model_name.startswith('tpmn') or model_name.startswith('lstm') or model_name.startswith('auto'):
                train_auc = train_tpmn(model, optimizer, train_data_loader, valid_data_loader, criterion, criterion2, criterion3, device, model_name)
                auc = test_tpmn(model, valid_data_loader, device, model_name)
            else: # 기존 모델
                train(model, optimizer, train_data_loader, criterion, device)
                auc = test(model, valid_data_loader, device)
            print('epoch:', epoch_i, 'train: auc:', train_auc)
            print('epoch:', epoch_i, 'validation: auc:', auc)
            
            train_auc_list.append(train_auc)
            auc_list.append(auc)

            if auc >= best_auc:
                best_auc = auc
                best_epoch = epoch_i
                torch.save(model, f'{save_dir}/{model_name}.pt')
            elif count == 0:
                count += 1
            elif count == 10:
                break


    
    model = torch.load(f'{save_dir}/{model_name}.pt')

    if model_name.startswith('tpmn') or model_name.startswith('lstm') or model_name.startswith('auto'):
        auc = test_tpmn(model, test_data_loader, device, model_name)
    else :
        auc = test(model, test_data_loader, device)
    
    for i in range(len(train_auc_list)) :
        print(i, 'epoch \t train auc : ', train_auc_list[i], '\t validation auc : ', auc_list[i])
    
    print('best epoch:', best_epoch) # validset
    print('best auc:', best_auc) # validset
    print('test auc:', auc)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo') # tpmn
    parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat') # dataset/tpmn_june_10_sample
    parser.add_argument('--model_name', default='afi') # tpmn_fm, tpmn_dfm, lstm_fm, lstm_dfm
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001) # 1/10로 줄이기
    parser.add_argument('--batch_size', type=int, default=4096) # 1024
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
