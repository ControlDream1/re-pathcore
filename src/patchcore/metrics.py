"""Anomaly metrics."""
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    计算图像级别的检索指标 (AUROC, FPR, TPR)。

    Args:
        anomaly_prediction_weights: [np.array 或 list] [N] 
                                    每张图像的异常预测分数。分数越高，表示是异常图像的概率越大。
        anomaly_ground_truth_labels: [np.array 或 list] [N] 
                                    二值化的真实标签。1 表示图像是异常，0 表示正常。
    
    Returns:
        dict: 包含 AUROC, FPR, TPR 和阈值的字典。
    """
    # 使用 scikit-learn 的 roc_curve 函数计算 ROC 曲线数据点（FPR 和 TPR）
    # fpr: False Positive Rate (假正率)
    # tpr: True Positive Rate (真正率)
    # thresholds: 用于计算 FPR 和 TPR 的决策阈值
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    # 使用 scikit-learn 的 roc_auc_score 函数计算 AUROC 值
    # AUROC (Area Under the Receiver Operating Characteristic Curve) 是评估分类器性能的常用指标
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    
    return {"auroc": auroc, "fpr": fpr, "tpr": tpr, "threshold": thresholds}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks):
    """
    计算像素级别的检索指标 (AUROC, F1-Score, FPR, FNR)。

    Args:
        anomaly_segmentations: [list of np.arrays 或 np.array] [NxHxW] 
                               模型生成的异常分割图（分数图）。
        ground_truth_masks: [list of np.arrays 或 np.array] [NxHxW] 
                             预定义的真实分割掩码。
    
    Returns:
        dict: 包含各种像素级指标（AUROC、最优阈值下的FPR/FNR）的字典。
    """
    # 将列表形式的分割图堆叠成一个 NumPy 数组
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    # 将所有图像的分割图和掩码展平为一维数组，方便进行像素级别的计算
    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    # 计算 ROC 曲线数据点和像素级 AUROC
    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    # 计算 Precision-Recall 曲线，用于找到最优阈值
    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    
    # 计算每个阈值下的 F1 分数
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # out 和 where 参数用于避免除以零的错误
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    # 找到能使 F1 分数最大的那个阈值，作为最优阈值
    optimal_threshold = thresholds[np.argmax(F1_scores)]
    
    # 基于最优阈值，将异常分数转换为二值预测（0或1）
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    
    # 计算在最优阈值下的 FPR 和 FNR (False Negative Rate)
    # FPR: 错误地将正常像素预测为异常像素的比例
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    # FNR: 错误地将异常像素预测为正常像素的比例
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
    }