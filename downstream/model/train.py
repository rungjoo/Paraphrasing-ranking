# -*- coding: utf-8 -*-
from tqdm import tqdm
import os
import random
import torch
import torch.nn.functional as F

from dataset import AllDatasetLoader
from model import EngModel, KorModel

from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

import pdb
import argparse, logging

from sklearn.metrics import precision_recall_fscore_support
    
## finetune DialoGPT
def main():
    """Dataset Loading"""
    dataset = args.dataset    
    if dataset == 'hate_speech18':
        train_path = '../paraphrase/'+dataset+'/'+dataset+'_train_split_norm.txt'
        dev_path = '../paraphrase/'+dataset+'/'+dataset+'_dev_split_norm.txt'
        test_path = '../paraphrase/'+dataset+'/'+dataset+'_test_split_norm.txt'
    else:
        train_path = '../paraphrase/'+dataset+'/'+dataset+'_train_split.txt'
        dev_path = '../paraphrase/'+dataset+'/'+dataset+'_dev_split.txt'
        test_path = '../paraphrase/'+dataset+'/'+dataset+'_test.txt'
        
    train_dataset = AllDatasetLoader(train_path, args.sample)    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)       
    
    dev_dataset = AllDatasetLoader(dev_path)
    dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    test_dataset = AllDatasetLoader(test_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print("training: {}, development: {}, test: {}".format(len(train_dataset), len(dev_dataset), len(test_dataset)))
    
    """logging and path"""
    if args.train_original:
        train_dataset_type = 'train_ori'
    else:
        train_dataset_type = 'train_para'
    if args.test_original:
        test_dataset_type = 'test_ori'
    else:
        test_dataset_type = 'test_para'
    if args.scratch:
        pretrained = 'scratch'
    else:
        pretrained = 'pretrained'
    save_path = os.path.join(dataset+'_models', pretrained, train_dataset_type+'_'+test_dataset_type)
    
    print("###Save Path### ", save_path)
    log_path = os.path.join(save_path, 'train.log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fileHandler = logging.FileHandler(log_path)
    
    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)    
    logger.setLevel(level=logging.DEBUG)      
    
    logger.info("Sampling: {}%".format(args.sample*100))
    
    """Model Loading"""
    if dataset == 'kor_hate':
        cls_num = 3
        model = KorModel(cls_num, args.scratch)
    elif dataset == 'financial':
        cls_num = 3
        model = EngModel(cls_num, args.scratch)
    elif dataset == 'hate_speech18':
        cls_num = 4 # 3
        model = EngModel(cls_num, args.scratch)
    model = model.cuda()    
    model.train() 
    
    """Training Setting"""    
    print('sctrach trainining?: ', args.scratch, '!!!')
    
    training_epochs = args.epoch
    save_term = int(training_epochs/5)
    max_grad_norm = args.norm
    lr = args.lr
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    if not args.train_original:
        num_training_steps *= 2
        num_warmup_steps *= 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    """Input & Label Setting"""
    best_dev_acc, best_test_acc = 0, 0
    best_dev_f1, best_test_f1 = 0, 0
    best_epoch = 0
    for epoch in tqdm(range(training_epochs)):
        model.train()
        for i_batch, (original, paraphrase, label) in enumerate(train_dataloader):
            original, paraphrase, label = original[0], paraphrase[0], label[0]
            label = torch.tensor([int(label)]).cuda()
                        
            """Prediction"""             
            original_idxs = model.tokenizer.encode(original, return_tensors='pt').cuda()
            original_idxs = original_idxs[:,:512]
            pred_outs = model(original_idxs)

            """Loss calculation & training"""
            loss_val = model.clsLoss(pred_outs, label)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            if not args.train_original:
                paraphrase_idxs = model.tokenizer.encode(paraphrase, return_tensors='pt').cuda()
                paraphrase_idxs = paraphrase_idxs[:,:512]
                pred_outs = model(paraphrase_idxs)

                """Loss calculation & training"""
                loss_val = model.clsLoss(pred_outs, label)

                loss_val.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
        """Dev & Test evaluation"""
        dev_acc, dev_pred_list, dev_label_list = _CalACC(model, dev_dataloader, args.test_original)
        test_acc, test_pred_list, test_label_list = _CalACC(model, test_dataloader, args.test_original)
        
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
            
        """Best Score & Model Save"""
        if args.f1:
            if epoch > 1 and dev_fbeta >= best_dev_f1:
                best_dev_f1 = dev_fbeta
                best_test_f1 = test_fbeta
                _SaveModel(model, save_path)
                best_epoch = epoch
            logger.info('Epoch: {}'.format(epoch))
            logger.info('Dev ## weighted-F1: {}'.format(dev_fbeta*100))
            logger.info('Test ## weighted-F1: {}'.format(test_fbeta*100))
        else:
            if epoch > 1 and dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                best_test_acc = test_acc
                _SaveModel(model, save_path)
                best_epoch = epoch        
            logger.info('Epoch: {}'.format(epoch))
            logger.info('Dev ## accuracy: {}'.format(dev_acc*100))
            logger.info('Test ## accuracy: {}'.format(test_acc*100))

    logger.info('Best Epoch: {}'.format(best_epoch))
    if args.f1:
        logger.info('Best Dev ## weighted-F1: {}'.format(dev_fbeta*100))
        logger.info('Sampling: {}%, Best Test ## weighted-F1: {}'.format(args.sample*100, best_test_f1*100))
    else:
        logger.info('Best Dev ## accuracy: {}'.format(best_dev_acc*100))
        logger.info('Sampling: {}%, Best Test ## accuracy: {}'.format(args.sample*100, best_test_acc*100))
    
def _CalACC(model, dataloader, original):
    eval_model = model.eval()
    pred_list = []
    label_list = []
    correct = 0
    with torch.no_grad():
        for i_batch, (original, paraphrase, label) in enumerate(dataloader):
            original, paraphrase, label = original[0], paraphrase[0], label[0]
            label = torch.tensor([int(label)]).cuda()
            """Prediction"""
            original_idxs = model.tokenizer.encode(original, return_tensors='pt').cuda()
            original_idxs = original_idxs[:,:512]
            pred_outs = eval_model(original_idxs)
            
            original_prob = F.softmax(pred_outs, 1)
            
            if not original:
                paraphrase_idxs = model.tokenizer.encode(paraphrase, return_tensors='pt').cuda()
                paraphrase_idxs = paraphrase_idxs[:,:512]
                pred_outs_paraphrase = eval_model(paraphrase_idxs)
                paraphrase_prob = F.softmax(pred_outs_paraphrase, 1)
                total_prob = original_prob + paraphrase_prob
            else:
                total_prob = original_prob
            
            """Calculation"""
            pred_label = total_prob.argmax(1)
            if pred_label == label:
                correct += 1
            pred_list.append(pred_label.item())
            label_list.append(label.item())
        acc = correct/len(dataloader)
    return acc, pred_list, label_list
        
def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'model.bin'))
    

if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    """Parameters"""
    parser  = argparse.ArgumentParser(description = "Downstream Classifier" )
    parser.add_argument( "--epoch", type=int, help = 'training epohcs', default = 10) # 12 for iemocap
    parser.add_argument( "--norm", type=int, help = "max_grad_norm", default = 10)
    parser.add_argument( "--lr", type=float, help = "learning rate", default = 1e-5) # 1e-5
    parser.add_argument( "--sample", type=float, help = "training dataset sampling", default = 1.0)

    parser.add_argument( "--dataset", help = 'financial or hate_speech18 or kor_hate', default = 'financial')
    parser.add_argument('--scratch', action='store_true', help='training from scratch')
    parser.add_argument('--f1', action='store_true', help='evalute acc or f1')
    parser.add_argument('--train_original', action='store_true', help='training dataset: original or paraphrase')
    parser.add_argument('--test_original', action='store_true', help='testing dataset: original or paraphrase')
        
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    streamHandler = logging.StreamHandler()
    
    main()
    