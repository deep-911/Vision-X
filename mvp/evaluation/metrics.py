import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, roc_auc_score, average_precision_score, brier_score_loss

def calculate_classification_metrics(y_true, y_pred_probs, threshold=2):
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 5-class metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], weights='quadratic')
    
    # Referable DR (Grade >= 2)
    ref_true = (y_true >= 2).astype(int)
    ref_probs = np.sum(y_pred_probs[:, 2:], axis=1)
    
    if isinstance(threshold, float):
        ref_pred = (ref_probs >= threshold).astype(int)
    else:
        ref_pred = (y_pred >= threshold).astype(int)
    
    ref_sens = recall_score(ref_true, ref_pred, zero_division=0)
    # Specificity = TN / (TN + FP)
    cm_binary = confusion_matrix(ref_true, ref_pred, labels=[0, 1])
    tn, fp, fn, tp = cm_binary.ravel()
    ref_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ref_roc = roc_auc_score(ref_true, ref_probs)
    ref_prc = average_precision_score(ref_true, ref_probs)
    
    # Calibration (Brier score for referable as example)
    brier = brier_score_loss(ref_true, ref_probs)
    
    return {
        'accuracy': acc,
        'balanced_accuracy': bacc,
        'macro_f1': macro_f1,
        'per_class_precision': prec.tolist(),
        'per_class_recall': rec.tolist(),
        'qwk': qwk,
        'confusion_matrix': cm.tolist(),
        'referable_sensitivity': ref_sens,
        'referable_specificity': ref_spec,
        'referable_roc_auc': ref_roc,
        'referable_pr_auc': ref_prc,
        'brier_score': brier
    }
