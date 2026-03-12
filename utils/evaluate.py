import numpy as np
import torch
from utils.logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


def evaluate(embeds, idx_train, idx_val, idx_test, label, nb_classes, device, lr, wd, log_dir,test_dataset,ratio,isTest=True):
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    train_embs = embeds[idx_train]
    val_embs = embeds[idx_val]
    test_embs = embeds[idx_test]

    train_lbls = torch.argmax(label[idx_train], dim=-1)
    val_lbls = torch.argmax(label[idx_val], dim=-1)
    test_lbls = torch.argmax(label[idx_test], dim=-1)
    accs = []
    micro_f1s = []
    macro_f1s = []
    macro_f1s_val = []
    auc_score_list = []

    for _ in range(5):
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)
        log.to(device)

        val_accs = []
        test_accs = []
        val_micro_f1s = []
        test_micro_f1s = []
        val_macro_f1s = []
        test_macro_f1s = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            # val
            logits = log(val_embs)
            preds = torch.argmax(logits, dim=1)

            val_acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
            val_f1_macro = f1_score(val_lbls.cpu(), preds.cpu(), average='macro')
            val_f1_micro = f1_score(val_lbls.cpu(), preds.cpu(), average='micro')

            val_accs.append(val_acc.item())
            val_macro_f1s.append(val_f1_macro)
            val_micro_f1s.append(val_f1_micro)

            # test
            logits = log(test_embs)
            preds = torch.argmax(logits, dim=1)

            test_acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
            test_f1_macro = f1_score(test_lbls.cpu(), preds.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds.cpu(), average='micro')

            test_accs.append(test_acc.item())
            test_macro_f1s.append(test_f1_macro)
            test_micro_f1s.append(test_f1_micro)
            logits_list.append(logits)

        max_iter = val_accs.index(max(val_accs))
        accs.append(test_accs[max_iter])
        max_iter = val_macro_f1s.index(max(val_macro_f1s))
        macro_f1s.append(test_macro_f1s[max_iter])
        macro_f1s_val.append(val_macro_f1s[max_iter])

        max_iter = val_micro_f1s.index(max(val_micro_f1s))
        micro_f1s.append(test_micro_f1s[max_iter])

        # auc
        best_logits = logits_list[max_iter]
        best_proba = softmax(best_logits, dim=1)
        auc_score = roc_auc_score(y_true=test_lbls.detach().cpu().numpy(),
                                  y_score=best_proba.detach().cpu().numpy(),
                                  multi_class='ovr'
                                  )
        auc_score_list.append(auc_score)
    
    if isTest:
        print("\t[Classification] Macro-F1: [{:.4f}, {:.4f}]  Micro-F1: [{:.4f}, {:.4f}]  auc: [{:.4f}, {:.4f}]"
              .format(np.mean(macro_f1s),
                      np.std(macro_f1s),
                      np.mean(micro_f1s),
                      np.std(micro_f1s),
                      np.mean(auc_score_list),
                      np.std(auc_score_list)
                      )
              )
        with open(log_dir, 'a') as f:
            f.write('\n')
            f.write(test_dataset)
            f.write(str(ratio) + ':' + str(np.mean(macro_f1s)) + '   '+ str(np.std(macro_f1s)) + "\t" + str(np.mean(micro_f1s))+ '   ' + str(np.std(micro_f1s)) + "\n")
            
        return np.mean(macro_f1s), np.mean(micro_f1s), np.mean(auc_score_list)
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)
    
    
import random

def evaluate_graph(embeds, mps, idx_train, idx_val, idx_test, label, nb_classes, device, lr, wd, log_dir, test_dataset, ratio, isTest=True):
    """
    1-shot per class evaluation:
        - For each class, randomly pick 1 training node.
        - Train LogReg on these nb_classes samples.
        - Evaluate on test set.
        - Repeat 100 times and average results.
    """
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    # ====== Pooling with adjacency matrix ======
    if mps.is_sparse:
        adj = mps.to_dense().to(device)
    else:
        adj = mps.to(device)

    adj_with_selfloop = adj + torch.eye(adj.size(0), device=device)
    degree = adj_with_selfloop.sum(1).clamp(min=1)
    degree_inv = 1.0 / degree
    norm_adj = adj_with_selfloop * degree_inv.unsqueeze(1)
    embeds_pooled = torch.mm(norm_adj, embeds.to(device))

    # Full label vectors
    full_labels = torch.argmax(label, dim=-1)  # [N]

    # Extract training labels and group by class
    train_nodes = idx_train.cpu().numpy()
    train_labels = full_labels[idx_train].cpu().numpy()

    # Build class -> list of train node indices (in idx_train)
    class_to_train_indices = {}
    for class_id in range(nb_classes):
        mask = (train_labels == class_id)
        nodes_in_class = np.where(mask)[0]  # indices in idx_train
        if len(nodes_in_class) == 0:
            raise ValueError(f"No training node for class {class_id}")
        class_to_train_indices[class_id] = nodes_in_class

    # Test data
    test_embs = embeds_pooled[idx_test]
    test_lbls = full_labels[idx_test]

    # Store results over 100 runs
    all_macro_f1s = []
    all_micro_f1s = []
    all_auc_scores = []

    num_runs = 100

    for run in range(num_runs):
        # Randomly select ONE node per class from training set
        selected_train_indices_in_idx_train = []
        for class_id in range(nb_classes):
            candidates = class_to_train_indices[class_id]
            chosen = np.random.choice(candidates)
            selected_train_indices_in_idx_train.append(chosen)

        # Map back to global node indices
        selected_global_indices = idx_train[selected_train_indices_in_idx_train]  # [C]

        # Get embeddings and labels
        train_embs_run = embeds_pooled[selected_global_indices]  # [C, D]
        train_lbls_run = full_labels[selected_global_indices]    # [C]

        # Train LogReg
        log = LogReg(hid_units, nb_classes).to(device)
        opt = torch.optim.Adam(log.parameters(), lr=lr, weight_decay=wd)

        # Train for few epochs (e.g., 100)
        for epoch in range(100):
            log.train()
            opt.zero_grad()
            logits = log(train_embs_run)
            loss = xent(logits, train_lbls_run)
            loss.backward()
            opt.step()

        # Evaluate on test set
        log.eval()
        with torch.no_grad():
            logits_test = log(test_embs)
            preds_test = torch.argmax(logits_test, dim=1)

            test_f1_macro = f1_score(test_lbls.cpu(), preds_test.cpu(), average='macro')
            test_f1_micro = f1_score(test_lbls.cpu(), preds_test.cpu(), average='micro')

            # AUC
            proba_test = softmax(logits_test, dim=1)
            try:
                auc_score = roc_auc_score(
                    y_true=test_lbls.cpu().numpy(),
                    y_score=proba_test.cpu().numpy(),
                    multi_class='ovr'
                )
            except ValueError:
                # In rare cases, predictions lack some classes
                auc_score = 0.0

            all_macro_f1s.append(test_f1_macro)
            all_micro_f1s.append(test_f1_micro)
            all_auc_scores.append(auc_score)

    # Final statistics
    mean_macro = np.mean(all_macro_f1s)
    std_macro = np.std(all_macro_f1s)
    mean_micro = np.mean(all_micro_f1s)
    std_micro = np.std(all_micro_f1s)
    mean_auc = np.mean(all_auc_scores)
    std_auc = np.std(all_auc_scores)

    if isTest:
        print("\t[1-Shot Per Class (100 runs)] Macro-F1: [{:.4f}, {:.4f}]  Micro-F1: [{:.4f}, {:.4f}]  AUC: [{:.4f}, {:.4f}]"
              .format(mean_macro, std_macro,
                      mean_micro, std_micro,
                      mean_auc, std_auc))
        with open(log_dir, 'a') as f:
            f.write('\n')
            f.write(test_dataset)
            f.write(str(ratio) + ':' + 
                    f"{mean_macro:.4f} ± {std_macro:.4f}\t" +
                    f"{mean_micro:.4f} ± {std_micro:.4f}\n")
        return mean_macro, mean_micro, mean_auc
    else:
        # For non-test mode, you might want val performance, but we didn't use val
        # Return dummy or compute val similarly
        return mean_macro, mean_macro