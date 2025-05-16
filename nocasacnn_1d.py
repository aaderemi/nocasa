import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import set_seed
from transformers import get_scheduler
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
import copy
import timm
import shutil
import torchio as tio
from skimage.metrics import structural_similarity
import random
import traceback
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import Wav2Vec2ForSequenceClassification
import librosa

wandb.login()

set_seed(42)
working_dir = ".."
sr = 44100
device = torch.device("cuda")

train_df = pd.read_csv(f"{working_dir}/train.csv")
train_idxs, test_idxs, ytr, ytest = train_test_split(train_df["File Name"],
                                               train_df.Score,
                                               random_state=42,
                                               stratify=train_df.Score, test_size=0.1)
words = train_df.Word.unique()
class CustomDataset2(torch.utils.data.Dataset):
    def __init__(self, idxs, base_dir):
        self.idxs = sorted(idxs)
        self.base_dir = base_dir
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, idx):
        file_idx = self.idxs[idx]
        mfcc = librosa.load(f"{self.base_dir}/{file_idx}", sr=None)
        score = train_df[train_df['File Name']==file_idx].Score.item()
        return (torch.tensor(mfcc[0]), torch.tensor(score-1, dtype = torch.long), file_idx)


trn_ds = CustomDataset2(list(train_idxs), "resampled_data")
val_ds = CustomDataset2(list(test_idxs), "resampled_data")

base_model = Wav2Vec2ForSequenceClassification.from_pretrained("NbAiLab/nb-wav2vec2-300m-nynorsk")
for p in base_model.wav2vec2.feature_extractor.parameters():
    p.requires_grad = False

class Wav2VecFeats(nn.Module):
    def __init__(self, base, p):
        super().__init__()
        self.base = copy.deepcopy(base)
        self.base.wav2vec2.encoder = nn.Identity()
        self.project = nn.Linear(512, 256)
        self.p = p
        self.dropout = nn.Dropout(self.p)
        self.classifier = nn.ModuleDict({word:nn.Sequential(nn.Linear(256,128),nn.ReLU(),self.dropout, nn.Linear(128,5)) for word in words})#nn.Linear(256, 5)
    def forward(self, x, fn):
        x = self.base.wav2vec2.feature_extractor(x)
        x = torch.mean(x, dim=-1)
        x = self.dropout(F.relu(self.project(x)))
        out = []
        for i,feat in enumerate(x):
            inp_word = train_df[train_df['File Name']==fn[i]].Word.values[0]
            pout = self.classifier[inp_word](feat)
            out.append(pout.squeeze())
        out = torch.vstack(out).to(x.device)
        return out


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def training(train_loader, model, criterion, optimizer, scheduler, device, scaler, accum_steps = 1, useauto = False):
    """
    Train a model.

    Args:
    - train_loader (torch.utils.data.DataLoader): The training data loader.
    - model (torch.nn.Module): The model to train.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimizer.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
    - device (torch.device): The device to run training on.

    Returns:
    - dict: A dictionary containing the average loss and accuracy.
    """
    print('\nTraining...')
    avg_meters = {"loss": AverageMeter(), "acc":AverageMeter()}

    model.train()
    model.to(device)
    model_ema = timm.utils.ModelEma(model, device=device)
    #print("Number of bacthes (training): ", len(train_loader))
    epoch_grad_norm = 0.0
    steps = 0
    autodtype = torch.float16 if useauto else torch.float32


    for batch_idx, (data, labels, fn) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)

        try:
            with torch.autocast(device_type=device.type, dtype=autodtype):
                output = model(data,fn)
                
                loss =   criterion(output, labels) # + criterion(output_ft, labels_ft)
            if torch.isnan(loss):
                print(f"\nNaN loss encountered in batch {batch_idx} !!!!")
                continue  # Skip this batch
            #loss.backward()
            scaler.scale(loss/accum_steps).backward()
            scale = scaler.get_scale()

            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) did not use gradient clipping while testing
            if (steps+1)%accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                if (scaler.get_scale() >= scale):
                    scheduler.step()  # Correct placement for OneCycleLR scheduler, inside the batch loop
                model_ema.update(model)
                #for p in model.parameters():
                #    epoch_grad_norm += torch.abs(p.grad).sum()
                optimizer.zero_grad(set_to_none=True)
                acc = (output.argmax(dim=-1) == labels).float().mean()
                
                avg_meters["loss"].update(loss.item(), data.size(0))
                avg_meters["acc"].update(acc.item(), data.size(0))

            
        except Exception as e:
            
            print(
                f"\nAn error occurred during training in batch {batch_idx}: {e}")
            print(traceback.format_exc())
            return

        
        steps += 1
        
        #wandb.log({"train_lossb": avg_meters['loss'].avg})
        #print(f'\nEnd of Batch {batch_idx} - Average Training Loss: {avg_meters["loss"].avg}')
    
    

    # print(f'\nEnd of Training - Average Loss: {avg_meters["loss"].avg}, Average Accuracy: {avg_meters["acc"].avg}')
    optimizer.zero_grad(set_to_none=True)
    return {"loss": avg_meters["loss"].avg, "acc":avg_meters["acc"].avg}, epoch_grad_norm

def validation(validation_loader, model, criterion, device, useauto=False):
    """
    Validate a model.

    Args:
    - validation_loader (torch.utils.data.DataLoader): The validation data loader.
    - model (torch.nn.Module): The model to validate.
    - criterion (torch.nn.Module): The loss function.
    - device (torch.device): The device to run validation on.

    Returns:
    - dict: A dictionary containing the average loss and accuracy.
    """
    print("\nValidating...")
    avg_meters = {"loss": AverageMeter(), "acc":AverageMeter()}

    model.eval()
    model.to(device)
    autodtype = torch.float16 if useauto else torch.float32
    #print("Number of bacthes (validation): ", len(validation_loader))

    try:
        with torch.no_grad():
            for batch_idx, (data, labels, fn) in enumerate(validation_loader):
                # No need to check for batch size if DataLoader is set to drop_last=True
                data, labels = data.to(device), labels.to(device)
                
                with torch.autocast(device_type=device.type, dtype=autodtype):
                    # Make prediction.
                    output = model(data,fn)

                    # Calculate the loss.
                    loss = criterion(output, labels)
                    acc = (output.argmax(dim=-1) == labels).float().mean()
                avg_meters["loss"].update(loss.item(), data.size(0))
                avg_meters["acc"].update(acc.item(), data.size(0))
                #wandb.log({"val_lossb": avg_meters['loss'].avg})
                #print(f'\nEnd of Batch {batch_idx} - Average Training Loss: {avg_meters["loss"].avg}')

        # print(f'\nAverage Validation Loss: {avg_meters["loss"].avg}, Average Validation Accuracy: {avg_meters["acc"].avg}')
        return {"loss": avg_meters["loss"].avg, "acc":avg_meters["acc"].avg}
    except Exception as e:
        print(f"\nAn error occurred during validation: {e}")
        print(traceback.format_exc())
        return {"loss": None, "acc": None}


def main_training_loop(train_func, val_func, num_epochs, device, run_name, loss, model, bs, lr, accum_steps, useauto=False):

    run = wandb.init(
          project="nocasa",
          config={
              "bs": bs,
              "num_epochs": num_epochs,
              "lr":lr
          },
          name=run_name
      )

    #criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    #criterion = nn.L1Loss()
    criterion = loss

    decay_list = []
    no_decay_list = []

    for n, m in model.named_parameters():
        if "bn" in n:
            no_decay_list.append(m)
        else:
            decay_list.append(m)

    param_list = [{"params":decay_list, "weight_decay":0.0}, {"params":no_decay_list, "weight_decay":0.0}]


    optimizer = torch.optim.Adam(param_list, lr= lr, betas = (0.9, 0.999))

    scheduler = get_scheduler("linear", optimizer, num_warmup_steps=50, num_training_steps=(num_epochs*len(trn_dl))/accum_steps)
    scaler = torch.amp.GradScaler(device.type)

    min_error = 100
    best_model = None
    grad_norms = []
    for epoch in tqdm(range(num_epochs)):
        

      train_results, epoch_grad_norm = train_func(trn_dl, model, criterion, optimizer, scheduler, device, scaler, accum_steps, useauto)
      val_results = val_func(val_dl, model, criterion, device, useauto)
      grad_norms.append(epoch_grad_norm)
      wandb.log({"train_loss": train_results['loss'], 
                    "val_loss": val_results['loss'],
                "train_acc":train_results["acc"], "val_acc":val_results["acc"]})


      print(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_results['loss']:.4f} Val Loss: {val_results['loss']:.4f} Train Acc: {train_results['acc']:.4f} Val Acc: {val_results['acc']:.4f}")

      if val_results['loss'] < min_error:
        min_error = val_results['loss']
        best_model = copy.deepcopy(model)
        torch.save(best_model.state_dict(), f"models/model_{run_name}.pth")

    torch.cuda.empty_cache()
    wandb.finish()
    return best_model, grad_norms

bs = 128
loss = nn.CrossEntropyLoss()
lr = 2e-3*(8/3)
accum_steps = 2

trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=bs, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=bs, shuffle=False)

mm = Wav2VecFeats(base_model, 0.5)

bm = main_training_loop(training, validation, 80, device, "cnn_1d_80_multi", loss, mm, bs, lr, accum_steps, useauto=True)

