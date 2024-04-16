import torch

def intersection_over_union(y_true, y_pred):
    intersection = torch.logical_and(y_true, y_pred).sum().float()
    union = torch.logical_or(y_true, y_pred).sum().float()
    iou_score = intersection / union
    return iou_score.item()

def dice_similarity_coefficient(y_true, y_pred):
    intersection = torch.sum(y_true * y_pred).float()
    dice_coefficient = (2. * intersection) / (torch.sum(y_true) + torch.sum(y_pred))
    return dice_coefficient.item()

def pixel_accuracy(y_true, y_pred):
    correct_pixels = torch.sum(y_true == y_pred).float()
    total_pixels = y_true.numel()
    accuracy = correct_pixels / total_pixels
    return accuracy.item()

def mean_pixel_accuracy(y_true, y_pred):
    class_accuracy = []
    for class_label in torch.unique(y_true):
        true_mask = (y_true == class_label)
        pred_mask = (y_pred == class_label)
        class_accuracy.append(torch.sum(true_mask == pred_mask).float() / torch.sum(true_mask))
    mean_accuracy = torch.mean(torch.stack(class_accuracy))
    return mean_accuracy.item()

def mean_intersection_over_union(y_true, y_pred):
    class_iou = []
    for class_label in torch.unique(y_true):
        true_mask = (y_true == class_label)
        pred_mask = (y_pred == class_label)
        class_iou.append(intersection_over_union(true_mask, pred_mask))
    mean_iou = torch.mean(torch.tensor(class_iou))
    return mean_iou.item()

def f1_score(y_true, y_pred):
    tp = torch.sum(y_true * y_pred).float()
    fp = torch.sum((1 - y_true) * y_pred).float()
    fn = torch.sum(y_true * (1 - y_pred)).float()
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1.item()





if __name__ == '__main__':
    # Example usage:
    y_true = torch.tensor([[0, 1, 1], [1, 0, 1]])
    y_pred = torch.tensor([[0, 1, 0], [1, 1, 1]])

    print("Intersection over Union:", intersection_over_union(y_true, y_pred))
    print("Dice Similarity Coefficient:", dice_similarity_coefficient(y_true, y_pred))
    print("Pixel Accuracy:", pixel_accuracy(y_true, y_pred))
    print("Mean Pixel Accuracy:", mean_pixel_accuracy(y_true, y_pred))
    print("Mean Intersection over Union:", mean_intersection_over_union(y_true, y_pred))
    print("F1 Score:", f1_score(y_true.flatten(), y_pred.flatten()))