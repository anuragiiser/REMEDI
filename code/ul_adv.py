from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import ast

import copy
from datetime import datetime
from utils import *
import argparse

parser = argparse.ArgumentParser(description="Gradient Ascent Unlearning (without retain set)")

parser.add_argument(
    "--model",
    type=str,
    required=True
)

parser.add_argument(
    "--forget_details_path",
    type=str,
    required=True,
    help="Path to forget set details JSON (e.g., data/forget_set.json)"
)

args = parser.parse_args()

MODEL = args.model

MODEL_PATH = "best_model/"+MODEL+"/"
UNLEARNED_MODEL_PATH = "unlearned/"+MODEL+"/adv/"
FORGET_DETAILS_PATH = args.forget_details_path

FORGET_LAMB = 0.3
RETAIN_LAMB = 0.7

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)

mlb, classes = get_classes_mlb()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
org_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

test_loader, val_loader, forget_details = load_val_test_loaders(FORGET_DETAILS_PATH, tokenizer, mlb)


criterion = SafeWeightedBCEWithLogitsLoss()

for UNLEARN_K in forget_details:
    print("\n\nUNLEARN_K:", UNLEARN_K)
    start_time = datetime.now()
    retain_score_1 = 0
    forget_score_1 = 0
    test_score_1 = 0
    # retain_score_2 = 0
    # forget_score_2 = 0
    # test_score_2 = 0
    model_dist = 0
    org_forget_score_1 = 0
    org_retain_score_1 = 0
    # org_forget_score_2 = 0

    for fold in forget_details[UNLEARN_K]:
        print("\n",fold)
        least_forget_score = 1.0
        unlearned_model_path = UNLEARNED_MODEL_PATH+UNLEARN_K+"/"+fold
        forget_loader, retain_loader, df_forget, df_retain = load_retain_forget_loaders(forget_details[UNLEARN_K][fold], tokenizer, mlb)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model = nn.DataParallel(model)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=UNLEARN_LR)

        adv_embs, adv_labels, adv_attn_masks = adv_attack(model, forget_loader, device)
        adv_dataset = AdvDataset(adv_embs, adv_labels, adv_attn_masks)

        adv_loader = DataLoader(
                adv_dataset,
                batch_size=NUM_ADV_SAMPLES*BATCH_SIZE,
                shuffle=True,
            )

        forget_metrics = test(model, forget_loader, device)
        retain_metrics = test(model, retain_loader, device)
        # print(f"Forget set before unlearning- Score 1: {forget_metrics["score_1"]: .4f} Score 2: {forget_metrics["score_2"]: .4f}")
        print(f"Forget set score (before unlearning): {forget_metrics["score_1"]: .4f}")
        print(f"Retain set score (before unlearning): {retain_metrics["score_1"]: .4f}")
        org_forget_score_1 += forget_metrics["score_1"]
        org_retain_score_1 += retain_metrics["score_1"]
        # org_forget_score_2 += forget_metrics["score_2"]

        for epoch in range(EPOCHS):
            # Training phase
            model.train()
            total_adv_loss = 0
            total_forget_loss = 0
            total_batches = len(forget_loader)
            samples = 0
            # progress_bar = tqdm(enumerate(zip(forget_loader, adv_loader)), desc=f'Epoch {epoch + 1}/{EPOCHS}')
            for i, (forget_batch, adv_batch) in enumerate(zip(forget_loader, adv_loader)):
                forget_input_ids = forget_batch['input_ids'].to(device)
                forget_attention_mask = forget_batch['attention_mask'].to(device)
                forget_labels = forget_batch['labels'].to(device)
                adv_embeds = adv_batch['embeds'].to(device)
                adv_attention_mask = adv_batch['attention_mask'].to(device)
                adv_labels = adv_batch['labels'].to(device)

                optimizer.zero_grad()
                forget_outputs = model(input_ids=forget_input_ids, attention_mask=forget_attention_mask)
                adv_outputs = model(inputs_embeds=adv_embeds, attention_mask=adv_attention_mask)

                
                # Check for NaN in model outputs
                # if torch.isnan(forget_outputs.logits).any() or torch.isnan(adv_outputs.logits).any():
                #     print(f"NaN detected in model outputs at batch {batch_count}")
                #     continue
                    
                # print(forget_labels, adv_labels)
                forget_loss = criterion(forget_outputs.logits, forget_labels)
                # Skip batch if loss is NaN
                # if torch.isnan(forget_loss):
                #     print(f"NaN loss detected at batch {batch_count}")
                #     continue

                adv_loss = criterion(adv_outputs.logits, adv_labels)
                # Skip batch if loss is NaN
                # if torch.isnan(adv_loss):
                #     print(f"NaN loss detected at batch {batch_count}")
                #     continue


                loss = RETAIN_LAMB*adv_loss - FORGET_LAMB*forget_loss
                # loss = -forget_loss + REG_LAMB*reg_loss
                # loss = -forget_loss
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                # scheduler.step()
                bs = len(forget_batch)
                total_adv_loss += adv_loss.item()*bs
                total_forget_loss += forget_loss.item()*bs
                samples += bs
                # progress_bar.set_postfix({
                #     'adv_loss': adv_loss.item(),
                #     'forget_loss': forget_loss.item(),
                #     'reg_loss': reg_loss.item()
                # })
                print(f"Batch {i+1}/{total_batches} Adv loss: {adv_loss.item()} Forget loss: {forget_loss.item()}")
            
            avg_adv_loss = total_adv_loss/samples
            avg_forget_loss = total_forget_loss/samples 
            
            print(f"\nEpoch {epoch + 1}\tAdv Loss: {avg_adv_loss:.4f}\tForget Loss: {avg_forget_loss: .4f}")
            forget_metrics = test(model, forget_loader, device)
            print(f"Epoch {epoch + 1} Forget set score: {forget_metrics["score_1"]:.4f}")
            if(forget_metrics["score_1"] < least_forget_score):
                least_forget_score = forget_metrics["score_1"]
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(unlearned_model_path)
                print("Model saved")
        
        model = AutoModelForSequenceClassification.from_pretrained(unlearned_model_path)
        model = nn.DataParallel(model)
        model.to(device)
        print(f"\nUnlearn K: {UNLEARN_K}\t {fold}")
        print(f"Forget set score: {least_forget_score:.4f}")
        retain_metrics = test(model, retain_loader, device)
        print(f"Retain set score: {retain_metrics["score_1"]:.4f}")
        test_metrics = test(model, test_loader, device)
        print(f"Test set score: {test_metrics["score_1"]:.4f}")
        dist = calculate_l2_distance(model, org_model, device)
        print(f"Model distance: {dist:.4f}")

        forget_score_1 += least_forget_score
        retain_score_1 += retain_metrics["score_1"]
        test_score_1 += test_metrics["score_1"]
        # forget_score_2 += forget_metrics["score_2"]
        # retain_score_2 += retain_metrics["score_2"]
        # test_score_2 += test_metrics["score_2"]
        model_dist += dist

        #model_to_save = model.module if hasattr(model, "module") else model
        #model_to_save.save_pretrained(UNLEARNED_MODEL_PATH+"_"+str(trial+1))
        #print("Model saved")

    end_time = datetime.now()
    delta = end_time - start_time
    total_seconds = int(delta.total_seconds()/TRIALS)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Unlearn_K: {UNLEARN_K}\tTotal Time Taken: {hours} hrs {minutes} mins {seconds} secs")

    avg_forget_score_1 = forget_score_1/TRIALS
    avg_retain_score_1 = retain_score_1/TRIALS
    avg_test_score_1 = test_score_1/TRIALS
    # avg_forget_score_2 = forget_score_2/TRIALS
    # avg_retain_score_2 = retain_score_2/TRIALS
    # avg_test_score_2 = test_score_2/TRIALS
    avg_model_dist = model_dist/TRIALS
    avg_org_forget_score_1 = org_forget_score_1/TRIALS
    avg_org_retain_score_1 = org_retain_score_1/TRIALS
    # avg_org_forget_score_2 = org_forget_score_2/TRIALS

    print(f"\nFinal scores of Unlearn_K: {UNLEARN_K}")
    print(f"Avg Forget score (before unlearning): {avg_org_forget_score_1:.4f}")
    print(f"Avg Retain score (before unlearning): {avg_org_retain_score_1:.4f}")
    print(f"Avg Forget Score: {avg_forget_score_1:.4f}")
    print(f"Avg Retain Score: {avg_retain_score_1:.4f}")
    print(f"Avg Test Score: {avg_test_score_1:.4f}")
    print(f"Avg Model Distance: {avg_model_dist:.4f}")