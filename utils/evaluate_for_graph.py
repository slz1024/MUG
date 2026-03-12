import numpy as np
import torch
from hgmae.utils.logreg import LogReg
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.functional import softmax
from sklearn.metrics import roc_auc_score

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score


def evaluate_cluster(embeds, y, n_labels, kmeans_random_state):
    Y_pred = KMeans(n_labels, random_state=kmeans_random_state).fit(embeds).predict(embeds)
    nmi = normalized_mutual_info_score(y, Y_pred)
    ari = adjusted_rand_score(y, Y_pred)
    return nmi, ari


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
    
    

def evaluate_graph(embeds, mps, idx_train, idx_val, idx_test, label, nb_classes, device, lr, wd, log_dir, test_dataset, ratio, isTest=True):
    """
    Args:
        embeds: [N, D] original node embeddings
        mps: adjacency matrix, [N, N], torch.Tensor (dense or sparse)
    """
    hid_units = embeds.shape[1]
    xent = nn.CrossEntropyLoss()

    if mps.is_sparse:
        adj = mps.to_dense().to(device)
    else:
        adj = mps.to(device)

    adj_with_selfloop = adj + torch.eye(adj.size(0), device=device)

    degree = adj_with_selfloop.sum(1).clamp(min=1)  
    degree_inv = 1.0 / degree 

    norm_adj = adj_with_selfloop * degree_inv.unsqueeze(1)  

    embeds_pooled = torch.mm(norm_adj, embeds.to(device)) 

    train_embs = embeds_pooled[idx_train]
    val_embs = embeds_pooled[idx_val]
    test_embs = embeds_pooled[idx_test]

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
            # Train
            log.train()
            opt.zero_grad()
            logits = log(train_embs)
            loss = xent(logits, train_lbls)
            loss.backward()
            opt.step()

            # Val
            log.eval()
            with torch.no_grad():
                logits_val = log(val_embs)
                preds_val = torch.argmax(logits_val, dim=1)
                val_acc = (preds_val == val_lbls).float().mean()
                val_f1_macro = f1_score(val_lbls.cpu(), preds_val.cpu(), average='macro')
                val_f1_micro = f1_score(val_lbls.cpu(), preds_val.cpu(), average='micro')

                val_accs.append(val_acc.item())
                val_macro_f1s.append(val_f1_macro)
                val_micro_f1s.append(val_f1_micro)

                # Test
                logits_test = log(test_embs)
                preds_test = torch.argmax(logits_test, dim=1)
                test_acc = (preds_test == test_lbls).float().mean()
                test_f1_macro = f1_score(test_lbls.cpu(), preds_test.cpu(), average='macro')
                test_f1_micro = f1_score(test_lbls.cpu(), preds_test.cpu(), average='micro')

                test_accs.append(test_acc.item())
                test_macro_f1s.append(test_f1_macro)
                test_micro_f1s.append(test_f1_micro)
                logits_list.append(logits_test)

        # Select best epoch based on validation metrics
        max_iter_acc = val_accs.index(max(val_accs))
        max_iter_macro = val_macro_f1s.index(max(val_macro_f1s))
        max_iter_micro = val_micro_f1s.index(max(val_micro_f1s))

        accs.append(test_accs[max_iter_acc])
        macro_f1s.append(test_macro_f1s[max_iter_macro])
        macro_f1s_val.append(val_macro_f1s[max_iter_macro])
        micro_f1s.append(test_micro_f1s[max_iter_micro])

        # AUC (use the same best epoch as micro/macro? Let's use macro for AUC to be consistent)
        best_logits = logits_list[max_iter_macro]
        best_proba = softmax(best_logits, dim=1)
        auc_score = roc_auc_score(
            y_true=test_lbls.cpu().numpy(),
            y_score=best_proba.cpu().numpy(),
            multi_class='ovr'
        )
        auc_score_list.append(auc_score)

    if isTest:
        print("\t[Classification] Macro-F1: [{:.4f}, {:.4f}]  Micro-F1: [{:.4f}, {:.4f}]  AUC: [{:.4f}, {:.4f}]"
              .format(np.mean(macro_f1s), np.std(macro_f1s),
                      np.mean(micro_f1s), np.std(micro_f1s),
                      np.mean(auc_score_list), np.std(auc_score_list)))
        with open(log_dir, 'a') as f:
            f.write('\n')
            f.write(test_dataset)
            f.write(str(ratio) + ':' + 
                    f"{np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}\t" +
                    f"{np.mean(micro_f1s):.4f} ± {np.std(micro_f1s):.4f}\n")
        return np.mean(macro_f1s), np.mean(micro_f1s), np.mean(auc_score_list)
    else:
        return np.mean(macro_f1s_val), np.mean(macro_f1s)
