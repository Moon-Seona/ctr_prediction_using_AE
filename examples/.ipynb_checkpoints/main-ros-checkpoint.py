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

from torchfm.dataset.tpmn import TPMNDataset 

#### New model
# baseline
from torchfm.model.base_fm import FactorizationMachineModel as base_fm # fm
from torchfm.model.base_dfm import DeepFactorizationMachineModel as base_dfm # dfm
# category
from torchfm.model.tpmn_fm import FactorizationMachineModel as tpmn_fm # fm
from torchfm.model.tpmn_dfm import DeepFactorizationMachineModel as tpmn_dfm # dfm
# lstm
from torchfm.model.lstm_fm import FactorizationMachineModel as lstm_fm
from torchfm.model.lstm_dfm import DeepFactorizationMachineModel as lstm_dfm 
# reconstruct
from torchfm.model.auto_fm import FactorizationMachineModel as auto_fm
from torchfm.model.auto_dfm import DeepFactorizationMachineModel as auto_dfm 
# all category 
from torchfm.model.all_tpmn_fm import FactorizationMachineModel as all_tpmn_fm
from torchfm.model.all_tpmn_dfm import DeepFactorizationMachineModel as all_tpmn_dfm 
# all lstm
from torchfm.model.all_lstm_fm import FactorizationMachineModel as all_lstm_fm
from torchfm.model.all_lstm_dfm import DeepFactorizationMachineModel as all_lstm_dfm 
# all reconstruct
from torchfm.model.all_auto_fm import FactorizationMachineModel as all_auto_fm
from torchfm.model.all_auto_dfm import DeepFactorizationMachineModel as all_auto_dfm 

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


def get_model(name, dataset, column):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    #parameter
    field_dims = dataset.field_dims
    embed_dim = 16
    dropout = 0.7
    #print(field_dims)
    #print(dataset.new_dict)
    #baseline
    if name == 'base_fm': 
        return base_fm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim)
    elif name == 'base_dfm' : 
        return base_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, mlp_dims=(8, 8), dropout=dropout)
    #category
    elif name == 'tpmn_fm': 
        return tpmn_fm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, column=column)
    elif name == 'tpmn_dfm' : 
        return tpmn_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, mlp_dims=(8, 8), dropout=dropout, column=column)
    # lstm
    elif name == 'lstm_fm' :
        return lstm_fm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim)
    elif name == 'lstm_dfm': 
        return lstm_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, mlp_dims=(8, 8), dropout=dropout)
    # auto, reconstruct
    elif name == 'auto_fm' :
        return auto_fm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim)
    elif name == 'auto_dfm': 
        return auto_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, mlp_dims=(8, 8), dropout=dropout)
    # all category
    elif name == 'all_tpmn_fm' :
        return all_tpmn_fm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim)
    elif name == 'all_tpmn_dfm': 
        return all_tpmn_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, mlp_dims=(8, 8), dropout=dropout)
    # lstm
    elif name == 'all_lstm_fm' :
        return all_lstm_fm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim)
    elif name == 'all_lstm_dfm': 
        return all_lstm_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, mlp_dims=(8, 8), dropout=dropout)
    # auto, reconstruct
    elif name == 'all_auto_fm' :
        return all_auto_fm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim)
    elif name == 'all_auto_dfm': 
        return all_auto_dfm(field_dims, len(dataset.new_dict)+1, embed_dim=embed_dim, mlp_dims=(8, 8), dropout=dropout)
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

def train_tpmn(model, optimizer, data_loader, val_data_loader, criterion, criterion2, device, model_name, column, log_interval=1000): # criterion3, mse loss 사용 안함
    model.train()
    total_loss = 0
    total_recon_loss = 0
    #total_mse_loss = 0

    targets, predicts = list(), list()

    for i, (fields, target, additional_info) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        model.train()

        fields, target, additional_info = fields.to(device), target.to(device), additional_info.to(device)
        if model_name.startswith('tpmn') or model_name.startswith('base') or model_name.startswith('all_tpmn') or model_name.startswith('lstm') or model_name.startswith('all_lstm'):
            y = model(fields, additional_info, column)

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
        elif model_name.startswith('auto') : # column 불러와서 if 문으로 정리하자
            y, pred, hidden, encode, embed = model(fields, additional_info, column) # pred2, pred3, encode2, 
            
            loss = criterion(y, target.float())

            # choose one or all infomation
            vocab = pred.shape[-1] # pred2, pred3, 185, len(dataset.new_dict)+1 인데... 마지막꺼 불러올 방법은?
            
            if column == 'appbundle' :
                col = additional_info[:,:149].reshape(-1)
                pred = pred.view(-1, vocab)
            elif column == 'carrier' :
                col = additional_info[:,149:199].reshape(-1)
                pred = pred.view(-1, vocab)
            else : #make
                col = additional_info[:,199:].reshape(-1)
                pred = pred.view(-1, vocab)
            
            recon_loss = criterion2(pred, col) # criterion2(pred, appbundle) + criterion2(pred3, make)
            
            model.zero_grad()
            
            (loss + recon_loss).backward()

            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            #total_mse_loss += mse_loss.item()

            targets.extend(target.tolist())
            predicts.extend(y.tolist())

            if (i + 1) % log_interval == 0:
                model.eval()
                accuracy = (col == torch.max(pred, dim=1)[1]).float().sum() / col.size(0)
                #accuracy2 = (carrier == torch.max(pred2, dim=1)[1]).float().sum() / carrier.size(0)
                #accuracy3 = (make == torch.max(pred3, dim=1)[1]).float().sum() / make.size(0)
                print('    - loss:', total_loss / log_interval, total_recon_loss / log_interval, '    - accuracy: ', accuracy.item()) # , accuracy2.item(), , accuracy3.item() 
                total_loss = 0
                total_recon_loss = 0
        elif model_name.startswith('all_auto') :
            y, pred, hidden, pred2, hidden2, pred3, hidden3, encode, encode2, encode3, embed, embed2, embed3 = model(fields, additional_info,column)  
            
            loss = criterion(y, target.float())

            # choose one or all infomation
            vocab = pred.shape[-1] # pred2, pred3, 185, len(dataset.new_dict)+1 인데... 마지막꺼 불러올 방법은?
            
            appbundle = additional_info[:,:149].reshape(-1)
            pred = pred.view(-1, vocab)
            carrier = additional_info[:,149:199].reshape(-1)
            pred2 = pred2.view(-1, vocab)
            make = additional_info[:,199:].reshape(-1) 
            pred3 = pred3.view(-1, vocab)
            
            recon_loss = criterion2(pred, appbundle) + criterion2(pred2, carrier) + criterion2(pred3, make)
            
            model.zero_grad()
            
            (loss + recon_loss).backward()

            optimizer.step()
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()

            targets.extend(target.tolist())
            predicts.extend(y.tolist())

            if (i + 1) % log_interval == 0:
                model.eval()
                accuracy = (appbundle == torch.max(pred, dim=1)[1]).float().sum() / appbundle.size(0)
                accuracy2 = (carrier == torch.max(pred2, dim=1)[1]).float().sum() / carrier.size(0)
                accuracy3 = (make == torch.max(pred3, dim=1)[1]).float().sum() / make.size(0)
                print('    - loss:', total_loss / log_interval, total_recon_loss / log_interval, '    - accuracy: ', accuracy.item(), accuracy2.item(), accuracy3.item()) 
                total_loss = 0
                total_recon_loss = 0
        else:
            raise ValueError('wrong model name: ' + model_name)
            
    return roc_auc_score(targets, predicts)

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


def test_tpmn(model, data_loader, device, model_name, column):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target, additional_info in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target, additional_info = fields.to(device), target.to(device), additional_info.to(device)
            if model_name.startswith('tpmn') or model_name.startswith('base') or model_name.startswith('all_tpmn') or model_name.startswith('all_lstm') or model_name.startswith('lstm'):
                y = model(fields, additional_info,column)
            elif model_name.startswith('auto') :
                y, _, _, _, _ = model(fields, additional_info,column)
            elif model_name.startswith('all_auto') :
                y, _, _, _, _, _, _, _, _, _, _, _, _ = model(fields, additional_info, column)
            else:
                raise ValueError('wrong model name: ' + model_name)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
        
    fp_rate, tp_rate, thresholds = roc_curve(targets, predicts)
    print(classification_report(targets, predicts > thresholds[np.argmax(tp_rate - fp_rate)]))
    print(confusion_matrix(targets, predicts > thresholds[np.argmax(tp_rate - fp_rate)]))
    
    return roc_auc_score(targets, predicts)

def get_sort(data, sorted_split1, column) :
    category_data = np.array([row[0] for row in data])
    click_data = np.array([row[1] for row in data])
    additional_data = np.array([row[2] for row in data])
    
    index = np.arange(len(data)) # just for concat
    
    if column == 'appbundle' :
        sorted_index = np.argsort(category_data[:,1]) # carrier 4, 약 2000개
        sorted_category_data = category_data[sorted_index]
        sorted_click_data = click_data[sorted_index]
        sorted_additional_data = additional_data[sorted_index]
        t2 = [(sorted_category_data[i], sorted_click_data[i], sorted_additional_data[i]) for i in index]
        
        train_dataset = t2[:sorted_split1*8]
        
        t3 = t2[sorted_split1*8:]
        random.shuffle(t3)
        valid_dataset = t3[:sorted_split1]
        test_dataset = t3[sorted_split1:]
        
        return train_dataset, valid_dataset, test_dataset
        
    elif column == 'carrier' :
        sorted_index = np.argsort(category_data[:,4]) # carrier 4, 약 2000개
        sorted_category_data = category_data[sorted_index]
        sorted_click_data = click_data[sorted_index]
        sorted_additional_data = additional_data[sorted_index]
        t2 = [(sorted_category_data[i], sorted_click_data[i], sorted_additional_data[i]) for i in index]
        
        train_dataset = t2[:sorted_split1*7] + t2[sorted_split1*8:sorted_split1*9]
        
        t3 = t2[sorted_split1*7:sorted_split1*8] +t2[sorted_split1*9:]
        random.shuffle(t3)
        valid_dataset = t3[:sorted_split1]
        test_dataset = t3[sorted_split1:]
        
        return train_dataset, valid_dataset, test_dataset
    
    elif column == 'make' :
        sorted_index = np.argsort(category_data[:,12]) # carrier 4, 약 2000개
        sorted_category_data = category_data[sorted_index]
        sorted_click_data = click_data[sorted_index]
        sorted_additional_data = additional_data[sorted_index]
        t2 = [(sorted_category_data[i], sorted_click_data[i], sorted_additional_data[i]) for i in index]
        
        train_dataset = t2[:sorted_split1*2] + t2[sorted_split1*3:sorted_split1*4] + t2[sorted_split1*5:]
        
        t3 = t2[sorted_split1*2:sorted_split1*3] + t2[sorted_split1*4:sorted_split1*5]
        random.shuffle(t3)
        valid_dataset = t3[:sorted_split1]
        test_dataset = t3[sorted_split1:]
        
        return train_dataset, valid_dataset, test_dataset
        
    else : # none
        t2 = [(category_data[i], click_data[i], additional_data[i]) for i in index]
        
        train_dataset = t2[:sorted_split1*8]

        t3 = t2[sorted_split1*8:]
        random.shuffle(t3)
        valid_dataset = t3[:sorted_split1]
        test_dataset = t3[sorted_split1:]

        return train_dataset, valid_dataset, test_dataset

def get_auc(model, optimizer, epoch, train_data_loader, valid_data_loader, test_data_loader, criterion, criterion2, device, save_dir, model_name, column) :
        best_auc = 0.0
        best_epoch = -1
        train = True
        count = 0
        train_auc_list = []
        auc_list = []
        
        if train :
            for epoch_i in range(epoch):
                if model_name.startswith('tpmn') or model_name.startswith('lstm') or model_name.startswith('auto') or model_name.startswith('base') or model_name.startswith('all'):
                    train_auc = train_tpmn(model, optimizer, train_data_loader, valid_data_loader, criterion, criterion2, device, model_name, column)
                    auc = test_tpmn(model, valid_data_loader, device, model_name, column)
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
        # finish train, start test
        model = torch.load(f'{save_dir}/{model_name}.pt')

        if model_name.startswith('tpmn') or model_name.startswith('lstm') or model_name.startswith('auto') or model_name.startswith('base') or model_name.startswith('all'):
            auc = test_tpmn(model, test_data_loader, device, model_name, column)
        else :
            auc = test(model, test_data_loader, device)

        for i in range(len(train_auc_list)) :
            print(i, 'epoch \t train auc : ', train_auc_list[i], '\t validation auc : ', auc_list[i])

        print('model name:', model_name)
        print('best epoch:', best_epoch) # validset
        print('best auc:', best_auc) # validset
        print('test auc:', auc)
        

def main(dataset_name,
         dataset_path,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir,
         column): #model_name,

    device = torch.device(device)
    dataset = get_dataset(dataset_name, dataset_path, '.tpmn_train') 
    
    train_length = int(len(dataset) * 0.7)
    #valid_length = int(len(dataset) * 0.05)
    #test_length = int(len(dataset) * 0.05)
    test_length2 = len(dataset) - train_length #- valid_length - test_length
    train_dataset, test_dataset2 = torch.utils.data.random_split( #valid_dataset, test_dataset, 
        dataset, (train_length, test_length2)) #valid_length, test_length, 
    
    data = np.array(train_dataset+test_dataset2) 
    
    category_data = data[:,0]
    click_data = data[:,1]
    additional_data = data[:,2]

    # click, nonclick 구분
    click = click_data[click_data == 1]
    click_category = category_data[click_data==1]
    click_additional = additional_data[click_data==1]
    
    nonclick = click_data[click_data == 0]
    nonclick_category = category_data[click_data==0]
    nonclick_additional = additional_data[click_data==0]
    
    # click data random 으로 nonclick 과 수 맞추기
    idx = np.arange(len(click))
    ros_idx = np.random.choice(idx, len(nonclick))
    
    ros_click = click[ros_idx]
    ros_category = click_category[ros_idx]
    ros_additional = click_additional[ros_idx]
    
    # 위에서 늘린 click data와 nonclick 데이터 합치기
    category = np.concatenate((ros_category, nonclick_category))
    click_total = np.concatenate((ros_click, nonclick))
    additional = np.concatenate((ros_additional, nonclick_additional))
    
    t = np.column_stack((category,click_total,additional))
    
    index = np.arange(len(t))

    # 그중에 random으로 일부 사용
    train_length = 10000000 # 천만
    #data_idx = np.random.choice(index, train_length) # 동일한 데이터 중복 선택 가능
    #t2 = t[data_idx]
    np.random.shuffle(t) # shuffle -> split
    t2 = t[:train_length] # error

    sorted_split1 = int(train_length * 0.1)
    train_dataset, valid_dataset, test_dataset = get_sort(t2, sorted_split1, column) 
    
    print('dataset length')
    print(len(train_dataset))
    print(len(valid_dataset))
    print(len(test_dataset))
    
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True) # batch 에서 shuffle 필요
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=0)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.CrossEntropyLoss()
    
    if column == 'none' : # base는 아무것도 포함안함, 나머지 6개 모델은 셋다 포함해서 진행
        model_list = ['base_fm', 'base_dfm']
        #, 'all_tpmn_fm', 'all_tpmn_dfm', 'all_lstm_fm', 'all_lstm_dfm', 'all_auto_fm', 'all_auto_dfm'
        for i in model_list :
            model = get_model(i, dataset, column).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            get_auc(model, optimizer, epoch, train_data_loader, valid_data_loader, test_data_loader, criterion, criterion2, device, save_dir, i, column)
    elif column == 'appbundle': #'base_fm', 'base_dfm', 
        model_list = ['tpmn_fm', 'auto_fm', 'tpmn_dfm', 'auto_dfm', 'lstm_fm', 'lstm_dfm']
        #,'all_tpmn_fm', 'all_tpmn_dfm', 'all_lstm_fm', 'all_lstm_dfm', 'all_auto_fm', 'all_auto_dfm']
        for i in model_list :
            model = get_model(i, dataset, column).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            get_auc(model, optimizer, epoch, train_data_loader, valid_data_loader, test_data_loader, criterion, criterion2, device, save_dir, i, column)
    elif column=='carrier': #'tpmn_fm', 
        model_list = ['auto_fm', 'tpmn_dfm', 'auto_dfm', 'lstm_fm', 'lstm_dfm']
        for i in model_list :
            model = get_model(i, dataset, column).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            get_auc(model, optimizer, epoch, train_data_loader, valid_data_loader, test_data_loader, criterion, criterion2, device, save_dir, i, column)
    elif column=='make' : #'tpmn_fm',  
        model_list = ['auto_fm', 'auto_dfm', 'lstm_fm', 'lstm_dfm']
        for i in model_list :
            model = get_model(i, dataset, column).to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            get_auc(model, optimizer, epoch, train_data_loader, valid_data_loader, test_data_loader, criterion, criterion2, device, save_dir, i, column)
    else :
        print('error 419line')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='criteo') # tpmn
    parser.add_argument('--dataset_path', help='criteo/train.txt, avazu/train, or ml-1m/ratings.dat') # dataset/tpmn_june_10_sample
    #parser.add_argument('--model_name', default='afi') # tpmn_fm, tpmn_dfm, lstm_fm, lstm_dfm
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=0.001) # 0.0001 
    parser.add_argument('--batch_size', type=int, default=4096) # 1024
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    parser.add_argument('--column', default='none')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir,
         args.column) #args.model_name,