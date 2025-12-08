from operator import index
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import ast
import copy
import json

SEED = 0
UNLEARN_LR = 1e-5
EPOCHS = 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 256
PGD_ALPHA = 0.001
PGD_ITER = 35
PGD_EPS = 0.15  
NUM_ADV_SAMPLES = 4
RETAIN_PER_FORGET = 4

TRIALS = 3
SEEDS = {"fold_1":42, "fold_2":43, "fold_3":44}

class SymptomICDDataset(Dataset):
    def __init__(self, symptoms, text, labels, tokenizer, max_length=512):
        self.symptoms = symptoms
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.symptoms)

    def __getitem__(self, idx):
        symptom_text = ' '.join(self.symptoms[idx])
        text = self.text[idx]
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
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }
    
def get_classes_mlb():
    # classes = str('403 486 582 585 425 276 710 724 458 287 285 275 583 558 327 228 338 789 790 V451 531 410 414 725 191 331 530 411 482 272 305 194 197 255 424 584 998 682 511 599 428 349 401 V100 V453 V586 041 251 E932 V300 V053 V290 571 070 250 570 572 286 518 038 280 263 995 303 244 112 881 903 955 E956 745 762 441 496 447 440 997 274 427 V104 V101 V120 V090 569 560 491 V458 433 436 493 996 416 V310 765 769 774 770 747 776 772 362 198 V103 746 766 V293 853 780 E888 730 357 430 293 443 V158 396 365 135 311 E935 721 214 437 242 600 189 304 711 800 E814 873 781 378 951 767 431 294 042 V141 V071 764 775 969 295 E950 266 779 355 553 965 E850 E853 426 804 E916 202 V502 398 707 348 787 564 V428 238 300 788 332 V107 V433 E879 861 423 E966 200 555 771 270 335 723 079 851 807 864 865 860 413 782 V108 507 512 752 162 783 778 333 785 136 799 E931 157 574 568 E878 722 719 V125 296 478 V170 805 596 E880 822 733 578 459 438 008 V098 185 967 225 V457 389 412 593 345 201 515 E933 278 492 715 415 V105 535 608 E870 V058 513 709 E821 V173 824 911 913 E812 576 203 281 580 V450 216 V340 579 693 351 088 714 E849 307 421 786 E942 959 E928 588 364 V642 V025 252 283 784 611 622 289 446 729 V498 V456 795 E854 V667 155 V130 882 852 957 E815 466 792 434 342 153 E934 481 910 456 453 867 273 532 806 V422 V541 556 394 444 924 E960 514 763 218 359 340 999 451 324 E939 537 737 455 E884 V427 591 592 577 557 575 356 368 552 500 750 253 292 E937 211 288 773 314 V652 432 379 435 E930 199 V641 494 966 758 E855 741 918 V436 078 562 820 801 839 E881 V584 731 E885 812 156 567 696 501 712 V707 215 754 753 508 876 720 V442 871 958 802 847 397 196 346 E968 510 404 360 376 370 V026 904 928 821 823 150 573 850 V497 E938 V533 V556 728 870 V874 V153 V644 V600 521 301 164 054 344 464 442 V150 282 V08 891 808 866 902 117 484 760 V048 691 519 528 320 369 685 V625 794 793 318 V441 761 936 E915 457 395 053 V113 V632 386 623 290 204 271 E819 811 813 884 E813 751 366 297 V440 473 E910 V420 057 536 152 970 485 235 372 E882 127 160 170 V880 595 909 V443 490 343 319 130 698 E823 246 854 868 872 982 151 V853 980 E980 291 517 268 487 E866 796 V452 036 354 648 701 V063 V038 227 614 533 736 942 E924 240 921 V454 977 759 768 923 E816 681 138 358 950 922 205 990 009 619 417 279 257 E860 755 991 E957 241 810 920 V461 V127 261 429 550 874 756 935 831 718 962 E858 803 480 674 277 880 879 377 529 047 083 835 462 336 E947 V160 420 317 454 E883 840 V550 960 586 933 597 350 E911 742 V614 298 V551 620 716 V462 V180 706 565 452 825 322 154 040 110 605 607 461 704 713 945 052 948 323 325 934 516 039 975 971 994 666 V111 907 E929 566 603 405 049 237 V161 V553 262 743 422 337 625 757 527 309 815 V163 402 869 E912 188 590 V852 V446 E852 886 E919 183 862 875 877 890 E944 E936 V444 598 V552 226 E818 617 E958 V123 748 968 V298 465 972 E826 905 E969 744 E829 V301 388 V146 V151 887 375 334 E848 E918 284 E876 260 987 E890 834 522 692 V588 310 863 E834 192 035 V174 171 738 220 477 212 172 V548 726 526 V099 777 749 E922 952 V320 901 542 449 V011 963 E822 524 V052 V539 144 445 321 380 604 383 587 137 845 695 V496 180 618 V102 540 525 916 174 V628 892 816 V171 520 708 176 791 V854 E906 V714 V554 V435 883 927 V434 007 581 V202 140 642 644 654 V270 V252 193 V838 V555 139 V195 V068 601 826 694 626 956 245 919 299 727 684 647 E941 V850 665 391 308 633 639 V230 V061 223 269 V183 046 534 361 673 643 986 005 034 382 239 232 V169 E901 908 634 836 616 E917 734 V698 133 E887 V445 V155 E949 142 E987 236 470 463 E940 229 448 702 182 E825 V851 814 V881 259 906 161 E891 830 E953 195 093 472 914 E988 930 543 686 900 075 705 939 381 V311 V168 018 004 917 483 656 641 217 V291 V164 E943 134 635 659 E920 506 E869 111 096 094 123 158 141 243 690 097 632 989 964 027 V596 373 V017 254 932 187 353 669 V504 602 843 912 374 983 E864 031 210 114 646 077 V018 670 615 V638 V135 938 V580 680 878 E965 471 652 663 658 V272 213 032 148 V643 V148 V062 E989 E927 131 233 V040 V066 125 V503 V581 V292 V192 700 703 209 V029 208 697 E871 184 015 146 V140 V154 992 249 149 V142 844 175 V542 363 V152 V106 V688 V265 012 885 E955 V530 385 V124 V741 390 474 627 817 230 E817 V198 E862 258 V463 735 V024 V640 976 E861 V765 V023 V626 E828 V188 341 V560 798 V448 893 495 084 523 V653 953 V549 V095 V182 621 475 V425 058 306 V165 551 E831 V136 V109 256 219 221 961 985 828 671 E820 897 V840 926 V421 048 594 896 082 E986 541 145 267 683 V097 732 265 011 E801 V185 664 V620 E840 V166 V468 629 115 V587 E908 120 V708 098 V469 V694 E824 E970 121 838 832 460 013 V239 944 V189 946 118 326 E945 645 352 159 E967 V618 147 V908 941 312 624 V186 V145 661 010 E865 091 E886 649 E905 E962 V612 E959 502 V438 V222 163 947 V162 E946 V716 315 367 V540 846 717 V561 V175 842 V138 V703 V583 841 672 062 488 347 339 E841 086 V400 E985 655 974 V289 V604 V074 V728 371 190 V126 090 143 943 V611 V331 085 V172 E835 668 740 V167 V558 E851 E811 V430 837 V072 V431 302 E923 V110 E900 V562 E963 E964 V118 V624 E800 988 833 023 V020 021 003 V660 E806 313 E954 V860 660 V449 231 V602 186 E863 E874 V721 V181 651 033 V654 E804 330 610 384 E838 E001 973 819 014 132 E899 925 207 V861 E002 E030 E000 894 E873 E999 E976 E003 V016 E805 045 V610 V078 V510 E029 848 E006 V403 122 V536 E013 E019 173 E913 677 E008 V568 V143 V091 V872 066 V601 116 V882 V065 538 V655 316 E007 E016 E921 V902 206 V254 099 V489 V870 E977 628 V250 E982 V486 539 V073 937 V812 030 V271 589 V672 V671 E926 E925 E857 V537 954 E827 657 V910 V789 V037 E975 V045 V848 393 V426 179 387 V903 E856 V901 915').split(' ')
    classes = ['401','427','428','276','250','414','272','285','518','584']
    print(f"Number of classes: {len(classes)}")

    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])
    return mlb, classes

def load_val_test_loaders(forget_details_path, tokenizer, mlb):

    with open(forget_details_path) as f:
        forget_details = json.load(f)

    df_test = pd.read_csv('data/mimic_test.csv')
    df_val = pd.read_csv('data/mimic_val.csv')

    df_test["icd9_codes"] = df_test["icd9_codes"].apply(ast.literal_eval)
    df_val['icd9_codes'] = df_val['icd9_codes'].apply(ast.literal_eval)

    print(f"Val Size: {len(df_val)}, Test Size: {len(df_test)}")

    test_labels = mlb.transform(df_test['icd9_codes'])
    val_labels = mlb.transform(df_val['icd9_codes'])

    test_dataset = SymptomICDDataset(
        df_test['Symptoms'],
        df_test['text'],
        test_labels,
        tokenizer
    )

    test_loader = DataLoader(
            test_dataset,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False
        )
    
    val_dataset = SymptomICDDataset(
        df_val['Symptoms'],
        df_val['text'],
        val_labels,
        tokenizer
    )

    val_loader = DataLoader(
            val_dataset,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False
        )

    return test_loader, val_loader, forget_details

def load_retain_forget_loaders(forget_inds, tokenizer, mlb):
    df = pd.read_csv('data/mimic.csv')
    df["icd9_codes"] = df["icd9_codes"].apply(ast.literal_eval)

    df_forget = df.loc[forget_inds].reset_index(drop=True)
    df_retain = df.drop(forget_inds).reset_index(drop=True)

    forget_labels = mlb.transform(df_forget['icd9_codes'])
    retain_labels = mlb.transform(df_retain['icd9_codes'])

    forget_dataset = SymptomICDDataset(
            df_forget['Symptoms'],
            df_forget['text'],
            forget_labels,
            tokenizer
        )

    forget_loader = DataLoader(
            forget_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
    
    retain_dataset = SymptomICDDataset(
            df_retain['Symptoms'],
            df_retain['text'],
            retain_labels,
            tokenizer
        )

    retain_loader = DataLoader(
            retain_dataset,
            batch_size=TEST_BATCH_SIZE,
            shuffle=False,
        )

    return forget_loader, retain_loader, df_forget, df_retain

# Modified loss function with safeguards
class SafeWeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, epsilon=1e-7):
        super().__init__()
        self.pos_weight = pos_weight
        self.epsilon = epsilon

    def forward(self, logits, targets):
        if self.pos_weight is None:
            pos_counts = targets.sum(dim=0)
            neg_counts = targets.size(0) - pos_counts
            # Add epsilon to avoid division by zero
            pos_weight = ((neg_counts + self.epsilon) / (pos_counts + self.epsilon)).clamp(min=1.0, max=100.0)
            self.pos_weight = pos_weight.to(logits.device)
        
        # Clip logits to prevent extreme values
        logits = torch.clamp(logits, min=-100, max=100)
        
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        
        # Check for NaN values
        if torch.isnan(loss).any():
            print("NaN detected in loss calculation!")
            print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")
            print(f"Target range: [{targets.min():.2f}, {targets.max():.2f}]")
            print(f"Pos weight range: [{self.pos_weight.min():.2f}, {self.pos_weight.max():.2f}]")
            
            # Replace NaN values with a high loss value
            loss = torch.nan_to_num(loss, nan=10.0)
        
        return loss.mean()
    

def compute_metrics(preds, labels):

    # predictions = torch.from_numpy(preds)
    # labels_t = torch.from_numpy(labels)
    # N, C = predictions.shape
    # k_per_row = labels_t.sum(dim=1)  # [N], number of positives

    # # argsort descending → ranks of each class
    # sorted_idx = predictions.argsort(dim=1, descending=True)           # [N, C]
    # ranks = torch.empty_like(sorted_idx)
    # ranks.scatter_(1, sorted_idx, torch.arange(C, device=predictions.device).expand(N, C))  # rank 0=best, C-1=worst

    # # expand k for comparison
    # k_exp = k_per_row.unsqueeze(1).expand(-1, C)

    # # mark top-k as 1
    # final_preds = (ranks < k_exp).int()

    # preds_2 = final_preds.numpy()
    preds_1 = (preds >= 0.5).astype(int)
    f1_macro_1 = f1_score(labels, preds_1, average='macro', zero_division=0)
    # f1_macro_2 = f1_score(labels, preds_2, average='macro', zero_division=0)
    
    return {
        'score_1': f1_macro_1,
        # 'score_2': f1_macro_2
    }

def test(model, loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Clip prediction values
            predictions = torch.sigmoid(torch.clamp(logits, min=-100, max=100))
            # predictions = torch.sigmoid(logits)
            
            # loss = criterion(logits, labels)
            # if not torch.isnan(loss):
            #     val_loss += loss.item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Calculate metrics with zero_division parameter
    return compute_metrics(
        np.array(all_predictions),
        np.array(all_labels)
    )

def calculate_l2_distance(model_a, model_b, device):
    model_a = model_a.module if hasattr(model_a, "module") else model_a
    model_b = model_b.module if hasattr(model_b, "module") else model_b
    model_a.to(device)
    model_b.to(device)
    total_l2 = 0.0
    with torch.no_grad():
        for (name_a, param_a), (name_b, param_b) in zip(model_a.named_parameters(), model_b.named_parameters()):
            assert name_a == name_b, f"Parameter mismatch: {name_a} != {name_b}"
            total_l2 += torch.norm(param_a - param_b, p=2).item() ** 2
    return total_l2 ** 0.5

criterion = SafeWeightedBCEWithLogitsLoss()

def pgd_attack_on_embeddings_batch_multilabel(
    model, embeddings, attention_mask, true_labels,
    num_adv_samples=NUM_ADV_SAMPLES, epsilon=PGD_EPS, alpha=PGD_ALPHA, num_steps=PGD_ITER, device="cuda"
):
    batch_size, num_classes = true_labels.size()

    # Expand embeddings, attention_mask, and labels
    embeddings = embeddings.unsqueeze(1).repeat(1, num_adv_samples, 1, 1)        # [B, N, L, D]
    attention_mask = attention_mask.unsqueeze(1).repeat(1, num_adv_samples, 1)  # [B, N, L]
    true_labels = true_labels.unsqueeze(1).repeat(1, num_adv_samples, 1)        # [B, N, C]

    embeddings = embeddings.view(batch_size * num_adv_samples, *embeddings.shape[2:])      # [B*N, L, D]
    attention_mask = attention_mask.view(batch_size * num_adv_samples, -1)                 # [B*N, L]
    true_labels = true_labels.view(batch_size * num_adv_samples, num_classes)              # [B*N, C]

    adv_targets = true_labels.clone().float().to(device) 

    n_samples, n_classes = true_labels.shape

    # mask of positives and negatives
    pos_mask = true_labels.bool()      # [N, C], True where label=1
    neg_mask = ~pos_mask               # True where label=0

    # generate random indices for each sample
    rand_pos = torch.randint(0, n_classes, (n_samples,), device=device)
    rand_neg = torch.randint(0, n_classes, (n_samples,), device=device)

    # make sure we pick actual positives for pos_idx
    for tries in range(20):  # up to 10 retries in batch
        invalid = ~pos_mask[torch.arange(n_samples), rand_pos]
        if not invalid.any():
            break
        rand_pos[invalid] = torch.randint(0, n_classes, (invalid.sum(),), device=device)

    # same for negatives
    for tries in range(20):
        invalid = ~neg_mask[torch.arange(n_samples), rand_neg]
        if not invalid.any():
            break
        rand_neg[invalid] = torch.randint(0, n_classes, (invalid.sum(),), device=device)

    # flip the chosen indices
    adv_targets[torch.arange(n_samples), rand_pos] = 0.0
    adv_targets[torch.arange(n_samples), rand_neg] = 1.0

    # ---- Clone embeddings ----
    adv_emb = embeddings.detach().clone().to(device)
    adv_emb.requires_grad = True

    # ---- PGD Iterations ----
    for _ in range(num_steps):
        outputs = model(inputs_embeds=adv_emb, attention_mask=attention_mask)  # ✅ pass attention mask
        logits = outputs.logits

        loss = criterion(logits, adv_targets)
        loss.backward()

        grad_sign = adv_emb.grad.detach().sign()
        adv_emb = adv_emb - alpha * grad_sign

        perturbation = torch.clamp(adv_emb - embeddings, min=-epsilon, max=epsilon)
        adv_emb = (embeddings + perturbation).detach()
        adv_emb.requires_grad = True

    # adv_emb = adv_emb.view(batch_size, num_adv_samples, *adv_emb.shape[1:])
    # adv_targets = adv_targets.view(batch_size, num_adv_samples, num_classes)

    return adv_emb.detach(), adv_targets.detach(), attention_mask.detach(), loss.item()

def pgd_attack_on_embeddings_batch_multilabel_modified(
    model, embeddings, attention_mask, true_labels, sampled_retain_labels,
    num_adv_samples=NUM_ADV_SAMPLES, epsilon=PGD_EPS, alpha=PGD_ALPHA, num_steps=PGD_ITER, device="cuda"
):
    batch_size, num_classes = true_labels.size()

    # Expand embeddings, attention_mask, and labels
    embeddings = embeddings.unsqueeze(1).repeat(1, num_adv_samples, 1, 1)        # [B, N, L, D]
    attention_mask = attention_mask.unsqueeze(1).repeat(1, num_adv_samples, 1)  # [B, N, L]
    true_labels = true_labels.unsqueeze(1).repeat(1, num_adv_samples, 1)        # [B, N, C]

    embeddings = embeddings.view(batch_size * num_adv_samples, *embeddings.shape[2:])      # [B*N, L, D]
    attention_mask = attention_mask.view(batch_size * num_adv_samples, -1)                 # [B*N, L]
    true_labels = true_labels.view(batch_size * num_adv_samples, num_classes)              # [B*N, C]

    rand_idx = torch.randint(0, len(sampled_retain_labels), (true_labels.size(0),), device=device)
    adv_targets = sampled_retain_labels[rand_idx]

    # ---- Clone embeddings ----
    adv_emb = embeddings.detach().clone().to(device)
    adv_emb.requires_grad = True

    # ---- PGD Iterations ----
    for _ in range(num_steps):
        outputs = model(inputs_embeds=adv_emb, attention_mask=attention_mask)  # ✅ pass attention mask
        logits = outputs.logits

        loss = criterion(logits, adv_targets)
        loss.backward()

        grad_sign = adv_emb.grad.detach().sign()
        adv_emb = adv_emb - alpha * grad_sign

        perturbation = torch.clamp(adv_emb - embeddings, min=-epsilon, max=epsilon)
        adv_emb = (embeddings + perturbation).detach()
        adv_emb.requires_grad = True

    return adv_emb.detach(), adv_targets.detach(), attention_mask.detach(), loss.item()

def adv_attack(model, unlearn_loader, device, sampled_retain_labels = None):
    model.eval()
    adv_embs = []
    adv_labels = []
    adv_attn_masks = []
    total_adv_loss = 0
    batch_count = 0
    print("Generating Adversial Embeddings")
    # progress_bar = tqdm(unlearn_loader)
    total_batches = len(unlearn_loader)
    for i, batch in enumerate(unlearn_loader):
        batch_count += 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Get embeddings for the whole batch
        embeddings = model.module.get_input_embeddings()(input_ids) if hasattr(model, "module") else model.get_input_embeddings()(input_ids)

        # Generate adversarial embeddings in batch
        if sampled_retain_labels is not None:
            adv_emb, adv_label, adv_attn_mask, adv_batch_loss = pgd_attack_on_embeddings_batch_multilabel_modified(model, embeddings, attention_mask, labels, sampled_retain_labels, device=device)
        else:
            adv_emb, adv_label, adv_attn_mask, adv_batch_loss = pgd_attack_on_embeddings_batch_multilabel(model, embeddings, attention_mask, labels, device=device)
        total_adv_loss += adv_batch_loss
        avg_adv_loss = total_adv_loss/batch_count
        # progress_bar.set_postfix({
        #     'avg_adv_loss': avg_adv_loss,
        # })
        adv_embs.extend(adv_emb.cpu().tolist())
        adv_labels.extend(adv_label.cpu().tolist())
        adv_attn_masks.extend(adv_attn_mask.cpu().tolist())
        print(f"Batch {i+1}/{total_batches} Avg_Loss: {avg_adv_loss}")

    return adv_embs, adv_labels, adv_attn_masks

class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, adv_embs, adv_labels, adv_attn_masks):
        self.embs = adv_embs
        self.labels = adv_labels
        self.attn_masks = adv_attn_masks

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        emb = self.embs[idx]
        label = self.labels[idx]
        mask = self.attn_masks[idx]
        return {"embeds": torch.tensor(emb, dtype=torch.float), "attention_mask": torch.tensor(mask, dtype=torch.long), "labels": torch.tensor(label, dtype=torch.float)}

def named_parameters_dict(model):
    """Return dict of {name: param} for parameters requiring grad.
       Works whether model is DataParallel or plain."""
    m = model.module if hasattr(model, "module") else model
    return {n: p for n, p in m.named_parameters() if p.requires_grad}

def parameter_regularization_loss(model, origin_params, importance):
    """Compute 0.5 * sum importance[n] * (p - origin[n])^2 over parameters."""
    reg = 0.0
    m_params = named_parameters_dict(model)
    for n, p in m_params.items():
        if n in importance:
            orig = origin_params[n].to(p.device)
            imp = importance[n].to(p.device)
            reg = reg + 0.5 * torch.sum(imp * (p - orig).pow(2))
    return reg

def estimate_parameter_importance(loader, model, device, num_samples=-1):
    """
    Estimate parameter importance by accumulating |grad| of L2-norm of model outputs.
    Returns dict {name: importance_tensor} matching named_parameters_dict(model).
    """
    # copy model to avoid changing original (we won't update its weights here)
    mcopy = copy.deepcopy(model.module if hasattr(model, "module") else model)
    mcopy.to(device)
    mcopy.train()

    params = {n: torch.zeros(p.shape, device=device) for n, p in mcopy.named_parameters() if p.requires_grad}

    # number of batches to use
    if num_samples is None or num_samples <= 0:
        n_batches = len(loader)
    else:
        n_batches = (num_samples + loader.batch_size - 1) // loader.batch_size

    # Use optimizer only to zero grads; no step
    opt = torch.optim.SGD(mcopy.parameters(), lr=1e-3)

    iter_loader = iter(loader)
    for _ in range(n_batches):
        try:
            batch = next(iter_loader)
        except StopIteration:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch["labels"].to(device)
        # forward
        outputs = mcopy(input_ids=input_ids, attention_mask=attention_mask)
        # Use logits L2-norm per sample, then mean
        logits = outputs.logits  # [B, C]
        # loss = torch.norm(logits, p=2, dim=1).mean()
        loss = criterion(logits, labels)
        opt.zero_grad()
        loss.backward()

        # accumulate abs gradients, scaled by batch size (like ref)
        bsz = input_ids.size(0)
        for n, p in mcopy.named_parameters():
            if p.grad is not None and n in params:
                params[n] += p.grad.detach().abs() * bsz

    # divide by total samples used
    total_samples = n_batches * loader.batch_size
    if total_samples == 0:
        total_samples = 1
    importance = {n: (p / total_samples).detach().clone() for n, p in params.items()}
    # Normalize each importance tensor to [0,1] then invert
    for n in list(importance.keys()):
        imp = importance[n]
        mn = imp.min()
        mx = imp.max()
        if mx - mn > 0:
            imp_norm = (imp - mn) / (mx - mn)
        else:
            imp_norm = torch.zeros_like(imp)
        # invert: high importance -> small allowed change, so regularizer weight = imp_norm (we want keep high imp)
        # reference used (1 - imp) later, but we'll keep imp_norm as importance (higher = more important)
        importance[n] = (1.0 - imp_norm).to(device)
    return importance

