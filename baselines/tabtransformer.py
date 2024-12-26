import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import pyreadstat
import pandas as pd
import pyreadstat
# from utils import EntireRowDataset
import sys

# Add path to the directory containing the models and utils modules
sys.path.append('/home/jyji/develop/Questionnaire-research/RAG/baselines')
from models.tabtrasformer import TabTransformer, TabularDataset2

import wandb

from sklearn.metrics import classification_report

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


def train_step(model, data_loader, optimizer, loss_fn, device, output_report=False):
    train_loss = 0.0
    train_acc = 0.0
    train_samples = 0

    y_trues = []
    y_preds = []

    model.train()
    for batch in tqdm(data_loader, ncols=80, desc='train step', leave=False):
        optimizer.zero_grad()

        # print(batch)

        cats, conts, labels = batch
        cats = cats.to(device)
        conts = conts.to(device)
        labels = labels.to(device)

        logits = model(cats, conts)
        loss = loss_fn(logits, labels)
        y_pred = logits.argmax(1)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * len(labels)
        train_acc += (y_pred == labels).sum().item()
        train_samples += len(labels)

        y_trues.extend(labels.cpu().tolist())
        y_preds.extend(y_pred.cpu().tolist())

    train_loss = train_loss / train_samples
    train_acc = train_acc / train_samples * 100.

    report = classification_report(y_trues, y_preds, zero_division=0, output_dict=True)

    if output_report:
        print(report)

    return report, train_loss


def valid_step(model, data_loader, loss_fn, device, output_report=False):
    valid_loss = 0.0
    valid_acc = 0.0
    valid_samples = 0

    y_trues = []
    y_preds = []

    model.eval()
    for batch in tqdm(data_loader, ncols=80, desc='valid step', leave=False):
        cats, conts, labels = batch
        cats = cats.to(device)
        conts = conts.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            logits = model(cats, conts)
        loss = loss_fn(logits, labels)
        y_pred = logits.argmax(1)

        valid_loss += loss.item() * len(labels)
        valid_acc += (y_pred == labels).sum().item()
        valid_samples += len(labels)

        y_trues.extend(labels.cpu().tolist())
        y_preds.extend(y_pred.cpu().tolist())

    valid_loss = valid_loss / valid_samples
    valid_acc = valid_acc / valid_samples * 100.

    report = classification_report(y_trues, y_preds, zero_division=0, output_dict=True)

    if output_report:
        print(report)

    return report, valid_loss

def collate_fn(samples):
    cats, conts, labels = [], [], []
    for sample in samples:
        cats.append(torch.LongTensor(sample[0]))
        conts.append(torch.FloatTensor(sample[1]))
        labels.append(sample[2])
    return (torch.stack(cats), torch.stack(conts), torch.tensor(labels))

def main(args):
    print(args)
    torch.manual_seed(args.seed)
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs

    best_valid_acc = 0.0
    best_epoch = -1
    history = []

    label_name = args.target_code
    train_dataset = TabularDataset2(label_name, train=True, user_ratio=0.5)
    eval_dataset = TabularDataset2(label_name, train=False, user_ratio=0.5)

    train_dataloder = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabTransformer(classes={col: train_dataset.df[col].unique().tolist() + [0] for col in train_dataset.df.columns if col not in ['THERMTRUMP_W116', 'THERMBIDEN_W116', label_name]},
                           cont_names=['THERMTRUMP_W116', 'THERMBIDEN_W116'],
                           c_out=args.c_out+1, #).to(device)
                           d_model=args.d_model,
                           n_layers=args.n_layers,
                           n_heads=args.n_heads,
                           mlp_mults=args.mlp_mults).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_report, train_loss = train_step(model, train_dataloder, optimizer, loss_fn, device)
        valid_report, valid_loss = valid_step(model, eval_dataloader, loss_fn, device, output_report=args.output_report)

        history.append({'epoch': epoch + 1,
        'train acc': train_report['accuracy'],
        'train precision': train_report['weighted avg']['precision'],
        'train recall': train_report['weighted avg']['recall'],
        'train f1': train_report['weighted avg']['f1-score'],
        'train loss': train_loss,
        'valid acc': valid_report['accuracy'],
        'valid precision': valid_report['weighted avg']['precision'],
        'valid recall': valid_report['weighted avg']['recall'],
        'valid f1': valid_report['weighted avg']['f1-score'],
        'valid loss': valid_loss,})

        print(f'Epoch {epoch+1} :\nTrain Acc : {train_report["accuracy"]:.2f}% | Train Loss : {train_loss:.4f} | Valid Acc : {valid_report["accuracy"]:.2f}% | Valid Loss : {valid_loss:.4f}')

        if valid_report['accuracy'] > best_valid_acc:
            best_valid_acc = valid_report['accuracy']
            best_epoch = epoch
            torch.save(model.state_dict(), f'weights/best_model_{args.target_code}{args.save_name}.pt')

    with open(f'history_{args.target_code}_50.json', 'w') as f:
        json.dump(history, f, indent=4)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Train hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--save_name', default='', type=str)
    parser.add_argument('--wandb', default=False, type=str2bool)
    parser.add_argument('--output_report', default=False, type=str2bool)

    # Model hyperparameters
    parser.add_argument('--year', default=2014, type=int)
    parser.add_argument('--target_code', default='SATIS_W116', type=str, required=True)
    parser.add_argument('--c_out', type=int, required=True)
    parser.add_argument('--d_model', default=32, type=int),
    parser.add_argument('--n_layers', default=6, type=int),
    parser.add_argument('--n_heads', default=8, type=int),
    parser.add_argument('--mlp_mults', nargs='+', default=[2], type=int),
    args = parser.parse_args()

    if args.wandb:
        wandb.init(
            project='pannel_lstm_token',
            group='TabTransformer',
            config=args,
        )
        wandb.config.update({
            'architecture': 'TabTransformer',
            'optimizer': 'AdamW',
            'dataset': 'HRS'
        })

    main(args)

    if args.wandb:
        wandb.finish()
