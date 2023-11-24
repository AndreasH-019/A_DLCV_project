import torch
import numpy as np
import copy

CLASSES = [22, 25]
def dice(outputs, targets, threshold):
    assert len(outputs) == len(targets)
    outputs, targets = copy.deepcopy(outputs), copy.deepcopy(targets)
    bs = len(outputs)
    dice_scores = {class_: [] for class_ in CLASSES}
    for i in range(bs):
        output, target = outputs[i], targets[i]
        keep = output['scores'] > threshold
        output['masks'] = output['masks'][keep]
        output['labels'] = output['labels'][keep]
        output['boxes'] = output['boxes'][keep]
        target_classes = torch.unique(target['labels'])
        for class_ in target_classes:
            new_output_mask = torch.any(output['masks'][output['labels'] == class_], dim=0)
            new_target_mask = torch.any(target['masks'][target['labels'] == class_], dim=0)
            intersection = torch.sum(torch.logical_and(new_output_mask, new_target_mask))
            combined_length = torch.sum(new_output_mask)+torch.sum(new_target_mask)+10**(-9)
            dice_score = 2*intersection/combined_length
            dice_scores[class_.item()].append(dice_score.item())


    dice_scores_meaned = {class_: np.mean(dice_scores[class_]) for class_ in CLASSES}
    return dice_scores_meaned