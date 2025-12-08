from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset, Dataset, DataLoader
import random
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import ast

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
UNLEARNED_MODEL_PATH = "unlearned/"+MODEL+"/scrub/"
FORGET_DETAILS_PATH = args.forget_details_path

start_time = datetime.now()
print("Start Time:", start_time)

LR_DECAY_EPOCHS = [1, 2, 3]
GAMMA = 0.6
ALPHA = 0.4
WEIGHT_DECAY = 0.1
M_STEPS = 1
KD_TEMP = 1.5
# RETAIN_PER_FORGET = 8

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

test_loader, val_loader, forget_details = load_val_test_loaders(FORGET_DETAILS_PATH, tokenizer, mlb)

criterion = SafeWeightedBCEWithLogitsLoss()

learned_teacher = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
learned_teacher = nn.DataParallel(learned_teacher)

# bce_logits = nn.BCEWithLogitsLoss(reduction='none')

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data, forget_labels, retain_labels, tokenizer, max_length=512):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_labels = forget_labels
        self.retain_labels = retain_labels
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, idx):
        if(idx < self.forget_len):
            data = self.forget_data.iloc[idx]
            labels = self.forget_labels[idx]
            y = 1
        else:
            data = self.retain_data.iloc[idx - self.forget_len]
            labels = self.retain_labels[idx - self.forget_len]
            y = 0
        symptom_text = ' '.join(data['Symptoms'])
        text = data['text']
        input_text = f"Symptoms: {symptom_text}\nNotes: {text}"
        
        inputs = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float),
            'y': y
        }

bce_logits = nn.BCEWithLogitsLoss(reduction='none')

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_t = torch.sigmoid(y_t/self.T)
        per_sample_per_class_loss = bce_logits(y_s/self.T, p_t)
        loss_per_sample = per_sample_per_class_loss.sum(dim=1)
        return loss_per_sample.mean()
        # p_s = F.log_softmax(y_s/self.T, dim=1)
        # p_t = F.softmax(y_t/self.T, dim=1)
        # loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        # return loss
    
criterion_cls = SafeWeightedBCEWithLogitsLoss()
criterion_div = DistillKL(KD_TEMP)

def unlearning_step(model, teacher, loader, optimizer, device, split="minimize"):
    model.train()
    total_loss = 0
    total_samples = 0
    total_batches = len(loader)
    for i, batch in enumerate(loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=input_ids, attention_mask=attention_mask).logits
        student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        loss_cls = criterion_cls(student_logits, labels)
        loss_div = criterion_div(student_logits, teacher_logits)

        if split == "minimize":
            loss = GAMMA * loss_cls + ALPHA * loss_div
        elif split == "maximize":
            loss = -loss_div
        loss = loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()*len(batch)
        total_samples += len(batch)
        print(f"Epoch {i+1}/{total_batches} Loss: {loss.item(): .4f}")
    return total_loss/total_samples, model

def adjust_learning_rate(epoch, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(LR_DECAY_EPOCHS))
    new_lr = UNLEARN_LR
    if steps > 0:
        new_lr = UNLEARN_LR * (WEIGHT_DECAY ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr

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

        df_sampled_retain = df_retain.sample(n=RETAIN_PER_FORGET*int(UNLEARN_K), random_state=SEEDS[fold]).reset_index(drop=True)
        sampled_retain_labels = mlb.transform(df_sampled_retain['icd9_codes'])

        sampled_retain_dataset = SymptomICDDataset(
                df_sampled_retain['Symptoms'],
                df_sampled_retain['text'],
                sampled_retain_labels,
                tokenizer
            )

        sampled_retain_loader = DataLoader(
                sampled_retain_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
        student = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        student = nn.DataParallel(student)
        optimizer = torch.optim.AdamW(student.parameters(), lr=UNLEARN_LR)

        learned_teacher.eval()
        total_batches = len(forget_loader)
        for epoch in range(EPOCHS):
            # lr = adjust_learning_rate(epoch, optimizer)
            maximize_loss = 0
            if(epoch <= M_STEPS): 
                maximize_loss, student = unlearning_step(model = student, loader = forget_loader, teacher=learned_teacher, 
                                                optimizer=optimizer, device=device, split="maximize")
            train_loss, student = unlearning_step(model = student, loader=sampled_retain_loader, teacher=learned_teacher, 
                                                optimizer=optimizer, device=device)
            print (f"Epoch {epoch+1}/{EPOCHS} maximize loss: {maximize_loss:.4f}\t minimize loss: {train_loss:.4f}")

            forget_metrics = test(student, forget_loader, device)
            print(f"Epoch {epoch + 1} Forget set score: {forget_metrics["score_1"]:.4f}")
            if(forget_metrics["score_1"] < least_forget_score):
                least_forget_score = forget_metrics["score_1"]
                model_to_save = student.module if hasattr(student, "module") else student
                model_to_save.save_pretrained(unlearned_model_path)
                print("Model saved")

        # student = AutoModelForSequenceClassification.from_pretrained(unlearned_model_path)
        # student = nn.DataParallel(student)
        # student.to(device)
        print(f"\nUnlearn K: {UNLEARN_K}\t {fold}")
        print(f"Forget set score: {forget_metrics["score_1"]:.4f}")
        retain_metrics = test(student, retain_loader, device)
        print(f"Retain set score: {retain_metrics["score_1"]:.4f}")
        test_metrics = test(student, test_loader, device)
        print(f"Test set score: {test_metrics["score_1"]:.4f}")
        dist = calculate_l2_distance(student, learned_teacher, device)
        print(f"Model distance: {dist:.4f}")

        forget_score_1 += forget_metrics["score_1"]
        retain_score_1 += retain_metrics["score_1"]
        test_score_1 += test_metrics["score_1"]
        # forget_score_2 += forget_metrics["score_2"]
        # retain_score_2 += retain_metrics["score_2"]
        # test_score_2 += test_metrics["score_2"]
        model_dist += dist

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