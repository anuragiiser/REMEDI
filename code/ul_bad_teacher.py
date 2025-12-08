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
from utils import *

from datetime import datetime
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

parser.add_argument(
    "--naive_model_path",
    type=str,
    required=True
)

args = parser.parse_args()

MODEL = args.model

MODEL_PATH = "best_model/"+MODEL+"/"
UNLEARNED_MODEL_PATH = "unlearned/"+MODEL+"/bad_teacher/"
FORGET_DETAILS_PATH = args.forget_details_path

NAIVE_MODEL_PATH = args.naive_model_path
KL_TEMP = 1

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

naive_teacher = AutoModelForSequenceClassification.from_pretrained(NAIVE_MODEL_PATH, num_labels=len(classes)).to(device)
naive_teacher = nn.DataParallel(naive_teacher)
learned_teacher = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
learned_teacher = nn.DataParallel(learned_teacher)

def UnlearnerLoss(student_logits, y, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    # labels = torch.unsqueeze(labels, dim = 1)
    y = y.float().unsqueeze(1)
    
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # f_teacher_probs = torch.sigmoid(full_teacher_logits / KL_temperature)
    # u_teacher_probs = torch.sigmoid(unlearn_teacher_logits / KL_temperature)

    # label 1 means forget sample
    # label 0 means retain sample
    # overall_teacher_probs = y * u_teacher_probs + (1.0-y)*f_teacher_probs
    overall_teacher_out = y * u_teacher_out + (1.0-y)*f_teacher_out
    student_out = F.log_softmax(student_logits / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)
    # per_sample_per_class_loss = bce_logits(student_logits / KL_temperature, overall_teacher_probs)
    # loss_per_sample = per_sample_per_class_loss.sum(dim=1)
    # if reduction == 'mean':
    #     loss = loss_per_sample.mean()
    # elif reduction == 'sum':
    #     loss = loss_per_sample.sum()
    # else:
    #     loss = loss_per_sample
    # return loss
    # student_out = F.log_softmax(output / KL_temperature, dim=1)
    # kl = F.kl_div(student_out, overall_teacher_out, reduction='batchmean')
    # return (KL_temperature**2) * kl
    # return F.kl_div(student_out, overall_teacher_out)

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
        forget_labels = mlb.transform(df_forget['icd9_codes'])

        student = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
        student = nn.DataParallel(student)
        optimizer = torch.optim.AdamW(student.parameters(), lr=UNLEARN_LR)

        unlearning_data = UnLearningData(df_forget, df_sampled_retain, 
                                    forget_labels, sampled_retain_labels, tokenizer)
        unlearning_loader = DataLoader(unlearning_data, batch_size = BATCH_SIZE, shuffle=True, 
                                    num_workers=0, pin_memory=True)
        

        forget_metrics = test(student, forget_loader, device)
        retain_metrics = test(student, retain_loader, device)
        # print(f"Forget set before unlearning- Score 1: {forget_metrics["score_1"]: .4f} Score 2: {forget_metrics["score_2"]: .4f}")
        print(f"Forget set score (before unlearning): {forget_metrics["score_1"]: .4f}")
        print(f"Retain set score (before unlearning): {retain_metrics["score_1"]: .4f}")
        org_forget_score_1 += forget_metrics["score_1"]
        org_retain_score_1 += retain_metrics["score_1"]
        # org_forget_score_2 += forget_metrics["score_2"]

        naive_teacher.eval()
        learned_teacher.eval()

        for epoch in range(EPOCHS):
            student.train()
            train_loss = 0
            samples = 0
            total_batches = len(unlearning_loader)
            for i, batch in enumerate(unlearning_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                # labels = batch['labels'].to(device)
                y = batch['y'].to(device)
                with torch.no_grad():
                    full_teacher_logits = learned_teacher(input_ids=input_ids, attention_mask=attention_mask).logits
                    unlearn_teacher_logits = naive_teacher(input_ids=input_ids, attention_mask=attention_mask).logits
                student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits

                optimizer.zero_grad()
                loss = UnlearnerLoss(student_logits = student_logits, y=y, full_teacher_logits=full_teacher_logits, 
                        unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_TEMP)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                batch_len = len(batch)
                train_loss += loss.item()*batch_len
                samples += batch_len
                print(f"Epoch {i+1}/{total_batches} Loss: {loss.item(): .4f}")
            
            avg_train_loss = train_loss/samples
            print(f"\nEpoch {epoch + 1} Avg Train Loss: {avg_train_loss:.4f}")

            forget_metrics = test(student, forget_loader, device)
            print(f"Epoch {epoch + 1} Forget set score: {forget_metrics["score_1"]:.4f}")
            if(forget_metrics["score_1"] < least_forget_score):
                least_forget_score = forget_metrics["score_1"]
                model_to_save = student.module if hasattr(student, "module") else student
                model_to_save.save_pretrained(unlearned_model_path)
                print("Model saved")
        
        student = AutoModelForSequenceClassification.from_pretrained(unlearned_model_path)
        student = nn.DataParallel(student)
        student.to(device)
        print(f"\nUnlearn K: {UNLEARN_K}\t {fold}")
        print(f"Forget set score: {least_forget_score:.4f}")
        retain_metrics = test(student, retain_loader, device)
        print(f"Retain set score: {retain_metrics["score_1"]:.4f}")
        test_metrics = test(student, test_loader, device)
        print(f"Test set score: {test_metrics["score_1"]:.4f}")
        dist = calculate_l2_distance(student, learned_teacher, device)
        print(f"Model distance: {dist:.4f}")

        forget_score_1 += least_forget_score
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