import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, cohen_kappa_score, roc_auc_score, average_precision_score, brier_score_loss

def calculate_classification_metrics(y_true, y_pred_probs):
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # 5-class metrics
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    prec = precision_score(y_true, y_pred, average=None)
    rec = recall_score(y_true, y_pred, average=None)
    cm = confusion_matrix(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    # Referable DR (Grade >= 2)
    ref_true = (y_true >= 2).astype(int)
    ref_probs = np.sum(y_pred_probs[:, 2:], axis=1)
    ref_pred = (y_pred >= 2).astype(int)
    
    ref_sens = recall_score(ref_true, ref_pred)
    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(ref_true, ref_pred).ravel()
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
