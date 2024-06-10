import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import config
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from save_results import (
    save_log,
    plot_loss,
)


def midpoint_to_corners(boxes):
    """
    Convert boxes from midpoint (x, y, w, h) to corner (x1, y1, x2, y2)

    Parameters:
        boxes (tensor): Bounding boxes (N, 4)

    Returns:
        tensor: Bounding boxes in corner format (N, 4)
    """

    x = boxes[..., 0:1]
    y = boxes[..., 1:2]
    w = boxes[..., 2:3]
    h = boxes[..., 3:4]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    convert_boxes = torch.cat((x1, y1, x2, y2), dim=-1)
    return convert_boxes

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(pred_boxes, true_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, best_mAP = 0, epoch = 0, mode = 'Train'):
    """
    Calculates mean average precision for multiple IoU thresholds.

    Parameters:
        pred_boxes (list): Bounding boxes predictions.
        true_boxes (list): Correct bounding boxes.
        iou_thresholds (list): IoU thresholds to evaluate.
        box_format (str): Format of bounding boxes, "midpoint" or "corners".
        num_classes (int): Number of classes.

    Returns:
        dict: mAP values for each IoU threshold.
    """
    mAPs = {}
    epsilon = 1e-6

    for iou_threshold in iou_thresholds:
        average_precisions = []

        for c in range(num_classes):
            detections = [detection for detection in pred_boxes if detection[1] == c]
            ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]

            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
                best_iou = 0

                for gt in ground_truth_img:
                    iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = ground_truth_img.index(gt)

                if best_iou > iou_threshold:
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            average_precisions.append(torch.trapz(torch.cat((torch.tensor([1]), precisions)), torch.cat((torch.tensor([0]), recalls))))

        mAPs[f'{iou_threshold}'] = sum(average_precisions) / (len(average_precisions) + epsilon)
        if mode == 'Valid' and iou_threshold == 0.5:
            if mAPs['0.5'] > best_mAP:
                from save_results import plot_AUC
                plot_AUC(precisions, recalls, mAPs['0.5'], epoch)        
    return mAPs['0.5'], mAPs['0.75'], mAPs['0.9']





def mean_average_precision_noise(pred_boxes, true_boxes, iou_thresholds=[0.5, 0.75, 0.9], box_format="midpoint", num_classes=config.NUM_CLASSES, mode = 'Train'):
    """
    Calculates mean average precision for multiple IoU thresholds.

    Parameters:
        pred_boxes (list): Bounding boxes predictions.
        true_boxes (list): Correct bounding boxes.
        iou_thresholds (list): IoU thresholds to evaluate.
        box_format (str): Format of bounding boxes, "midpoint" or "corners".
        num_classes (int): Number of classes.

    Returns:
        dict: mAP values for each IoU threshold.
    """
    mAPs = {}
    epsilon = 1e-6

    for iou_threshold in iou_thresholds:
        average_precisions = []

        for c in range(num_classes):
            detections = [detection for detection in pred_boxes if detection[1] == c]
            ground_truths = [true_box for true_box in true_boxes if true_box[1] == c]

            amount_bboxes = Counter([gt[0] for gt in ground_truths])

            for key, val in amount_bboxes.items():
                amount_bboxes[key] = torch.zeros(val)

            detections.sort(key=lambda x: x[2], reverse=True)
            TP = torch.zeros((len(detections)))
            FP = torch.zeros((len(detections)))
            total_true_bboxes = len(ground_truths)

            if total_true_bboxes == 0:
                continue

            for detection_idx, detection in enumerate(detections):
                ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
                best_iou = 0

                for gt in ground_truth_img:
                    iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[3:]), box_format=box_format)

                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = ground_truth_img.index(gt)

                if best_iou > iou_threshold:
                    if amount_bboxes[detection[0]][best_gt_idx] == 0:
                        TP[detection_idx] = 1
                        amount_bboxes[detection[0]][best_gt_idx] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1

            TP_cumsum = torch.cumsum(TP, dim=0)
            FP_cumsum = torch.cumsum(FP, dim=0)
            recalls = TP_cumsum / (total_true_bboxes + epsilon)
            precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
            average_precisions.append(torch.trapz(torch.cat((torch.tensor([1]), precisions)), torch.cat((torch.tensor([0]), recalls))))

        mAPs[f'{iou_threshold}'] = sum(average_precisions) / (len(average_precisions) + epsilon)     
    return mAPs['0.5'], mAPs['0.75'], mAPs['0.9']



def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()




def get_bboxes(
    epoch=0,
    loader=None,
    model=None,
    loss_fn=None,
    iou_threshold=0.5,
    threshold=0.4,
    pred_format="cells",
    box_format="midpoint",
    device="cpu",
    mode = "Train"
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    loop = tqdm(loader, leave=True)
    mean_loss = []
    mean_loss_box_coordinates = []
    mean_loss_object_loss = []
    mean_loss_no_object_loss = []
    mean_loss_class = []

    loop.set_description(f"Eval: {mode}: ")    
    for batch_idx, (x, labels) in enumerate(loop):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)
            loss, loss_box_coordinates, loss_object_loss, loss_no_object_loss, loss_class = loss_fn(predictions, labels)
            mean_loss.append(loss.item())   
            mean_loss_box_coordinates.append(loss_box_coordinates.item())
            mean_loss_object_loss.append(loss_object_loss.item())
            mean_loss_no_object_loss.append(loss_no_object_loss.item())
            mean_loss_class.append(loss_class.item())         

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            
            all_boxes = bboxes[idx]
            # same but considering all boxes, i.e.,
        
            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    if mode == 'Train':
        mean_loss = mean_loss[:len(mean_loss)//2] 
    # Resumen loss
    mean_loss_out = sum(mean_loss)/len(mean_loss)
    mean_loss_box_coordinates_out = sum(mean_loss_box_coordinates)/len(mean_loss_box_coordinates)
    mean_loss_object_loss_out = sum(mean_loss_object_loss)/len(mean_loss_object_loss)
    mean_loss_no_object_loss_out = sum(mean_loss_no_object_loss)/len(mean_loss_no_object_loss)
    mean_loss_class_out = sum(mean_loss_class)/len(mean_loss_class)

    print("---------------------------------------------------")
    print('-------------Loss Summary eval {mode}--------------')
    print("Total Loss".ljust(12) + "|" + 
          "Loss Coord".ljust(12) + "|" + 
          "Conf Loss".ljust(12) + "|" + 
          "No Obj Loss".ljust(12) + "|" +
            "Class Loss".ljust(12) + "|") 
    print(f'{mean_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_box_coordinates_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_object_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_no_object_loss_out:.3f}'.ljust(12) + "|"+
          f'{mean_loss_class_out:.3f}'.ljust(12) + "|")  
    print("---------------------------------------------------")

    if mode == 'Train' or mode == 'Valid':
        save_log(epoch, mean_loss_out, mean_loss_box_coordinates_out, mean_loss_object_loss_out, mean_loss_no_object_loss_out, mean_loss_class_out, mode='eval', eval_mode=mode)
        plot_loss(mode='eval', eval_mode=mode)
        model.train()

    # all pred boxes are for all images = [train_idx, class_pred, prob_score, x, y, w, h]
    # all true boxes are for all images = [train_idx, class_pred, prob_score, x, y, w, h]
    return all_pred_boxes, all_true_boxes, mean_loss_out

def get_bboxes_noise(
    loader,
    model,
    loss_fn,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cpu",
    mode="Train",
    add_noise_to_input=None,
    noise_level=0.1
):
    all_pred_boxes = []
    all_true_boxes = []

    # Make sure model is in eval mode before getting bboxes
    model.eval()
    train_idx = 0
    loop = tqdm(loader, leave=True)
    mean_loss = []
    mean_loss_box_coordinates = []
    mean_loss_object_loss = []
    mean_loss_no_object_loss = []
    mean_loss_class = []

    loop.set_description(f"Eval: {mode}: ")
    for batch_idx, (x, labels) in enumerate(loop):
        x = x.to(device)
        labels = labels.to(device)

        # Add noise to input if specified
        if add_noise_to_input:
            x = add_noise_to_input(x, noise_level)

        with torch.no_grad():
            predictions = model(x)
            loss, loss_box_coordinates, loss_object_loss, loss_no_object_loss, loss_class = loss_fn(predictions, labels)
            mean_loss.append(loss.item())
            mean_loss_box_coordinates.append(loss_box_coordinates.item())
            mean_loss_object_loss.append(loss_object_loss.item())
            mean_loss_no_object_loss.append(loss_no_object_loss.item())
            mean_loss_class.append(loss_class.item())

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    # Loss summary
    mean_loss_out = sum(mean_loss) / len(mean_loss)
    mean_loss_box_coordinates_out = sum(mean_loss_box_coordinates) / len(mean_loss_box_coordinates)
    mean_loss_object_loss_out = sum(mean_loss_object_loss) / len(mean_loss_object_loss)
    mean_loss_no_object_loss_out = sum(mean_loss_no_object_loss) / len(mean_loss_no_object_loss)
    mean_loss_class_out = sum(mean_loss_class) / len(mean_loss_class)

    print("---------------------------------------------------")
    print(f'-------------Loss Summary eval {mode}--------------')
    print("Total Loss".ljust(12) + "|" +
          "Loss Coord".ljust(12) + "|" +
          "Conf Loss".ljust(12) + "|" +
          "No Obj Loss".ljust(12) + "|" +
          "Class Loss".ljust(12) + "|")
    print(f'{mean_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_box_coordinates_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_object_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_no_object_loss_out:.3f}'.ljust(12) + "|" +
          f'{mean_loss_class_out:.3f}'.ljust(12) + "|")
    print("---------------------------------------------------")

    # all pred boxes are for all images = [train_idx, class_pred, prob_score, x, y, w, h]
    # all true boxes are for all images = [train_idx, class_pred, prob_score, x, y, w, h]
    return all_pred_boxes, all_true_boxes, mean_loss_out



def convert_cellboxes(predictions):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, config.SPLIT_SIZE, config.SPLIT_SIZE, config.NUM_CLASSES+config.NUM_BOXES*5)
    bboxes1 = predictions[..., -9:-5]
    bboxes2 = predictions[..., -4:]
    scores = torch.cat(
        (predictions[..., -10].unsqueeze(0), predictions[..., -5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(config.SPLIT_SIZE).repeat(batch_size, config.SPLIT_SIZE, 1).unsqueeze(-1)
    x = 1 / config.SPLIT_SIZE * (best_boxes[..., :1] + cell_indices)
    y = 1 / config.SPLIT_SIZE * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / config.SPLIT_SIZE * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :-10].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., -10], predictions[..., -5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], config.SPLIT_SIZE * config.SPLIT_SIZE, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(config.SPLIT_SIZE * config.SPLIT_SIZE):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def convert_cellboxes_test(predictions, SPLIT_SIZE, NUM_BOXES, NUM_CLASSES):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, SPLIT_SIZE, SPLIT_SIZE, NUM_CLASSES+NUM_BOXES*5)
    bboxes1 = predictions[..., -9:-5]
    bboxes2 = predictions[..., -4:]
    scores = torch.cat(
        (predictions[..., -10].unsqueeze(0), predictions[..., -5].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(SPLIT_SIZE).repeat(batch_size, SPLIT_SIZE, 1).unsqueeze(-1)
    x = 1 / SPLIT_SIZE * (best_boxes[..., :1] + cell_indices)
    y = 1 / SPLIT_SIZE * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / SPLIT_SIZE * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :-10].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., -10], predictions[..., -5]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes_test(out, SPLIT_SIZE, NUM_BOXES, NUM_CLASSES):
    converted_pred = convert_cellboxes_test(out, SPLIT_SIZE, NUM_BOXES, NUM_CLASSES).reshape(out.shape[0], SPLIT_SIZE * SPLIT_SIZE, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(SPLIT_SIZE * SPLIT_SIZE):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def find_the_best_model(folder_path):
    import re
    pattern = re.compile(r'(\d+)_YOLO_best\.pth\.tar$')
    
    largest_index = -1
    largest_file = None
    
    # Search for the largest index
    for filename in os.listdir(folder_path):
        match = pattern.search(filename)
        if match:
            # Extract index
            index = int(match.group(1))
            # Compare index
            if index > largest_index:
                largest_index = index
                largest_file = filename
    
    return largest_file, largest_index

def plot_batch_imgs(
    loader,
    iou_threshold=0.5,
    threshold=0.4,
    device="cpu",
    pred=False,
    num_plot=1
):
    assert True, "unfinished, deprecated"
    (pred_bboxes,true_bboxes,x) = get_batch_bboxes(
        loader=loader,
        iou_threshold=iou_threshold,
        threshold=threshold,
        device=device,
        pred=pred
    )

    batch_size = len(x)
    nimgs = num_plot if num_plot<=batch_size else batch_size

def get_loaders():
    from dataset import YOLOdataset

    train_dataset = YOLOdataset(
        transform=config.train_transforms,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        mode="train",
        S = config.SPLIT_SIZE,
        B = config.NUM_BOXES,
        C = config.NUM_CLASSES
    )
    valid_dataset = YOLOdataset(
        transform=config.valid_transforms,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        mode="valid",
        S = config.SPLIT_SIZE,
        B = config.NUM_BOXES,
        C = config.NUM_CLASSES
    )
    train_eval_dataset = YOLOdataset(
        transform=config.valid_transforms,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        mode="train",
        S = config.SPLIT_SIZE,
        B = config.NUM_BOXES,
        C = config.NUM_CLASSES
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,        
        shuffle=True,
        drop_last=False,
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,        
        shuffle=True,
        drop_last=False,
    )

    return train_loader, valid_loader, train_eval_loader

def get_test_loader():
    from dataset import YOLOdataset

    test_dataset = YOLOdataset(
        transform=config.test_transform,
        img_dir=config.IMG_DIR,
        label_dir=config.LABEL_DIR,
        mode="test",
        S = 100,
        B = 2,
        C = config.NUM_CLASSES
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    return test_loader

def extract_image_name(folder):
# Name of all the files that are inside the folder images
    images_names = []
    labels_names = []

    for files in os.listdir(folder):
        complete_path = os.path.join(folder, files)
        if os.path.isfile(complete_path) and files.endswith('.jpg'):
            images_names.append(files)
            labels_names.append(files.replace('.jpg', '.txt'))
    return images_names, labels_names


def seed_everything(seed=42):
    import random
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def warmup_scheduler(optimizer, epoch):
    target_lr = config.BASE_lr
    num_steps = config.WARM_UP
    lr_increment = (target_lr - config.INIT_lr) / num_steps
    if config.WARM_UP < epoch:
        for i in range(num_steps):
            lr = config.INIT_lr + i * lr_increment
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # Tu función de entrenamiento por una epoch


def update_lr(optimizer, epoch):
    if epoch == 0:
        lr = config.INIT_lr 
    elif epoch == 2:
        lr = config.BASE_lr
    elif epoch == 15:
        lr = config.BASE_lr/2
    elif epoch == 25:
        lr = config.INIT_lr
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def update_lr_no_pretrained(optimizer, epoch):
    init_value = config.INIT_lr
    final_value = config.BASE_lr
    num = config.WARM_UP
    if epoch >= 0 and epoch < config.WARM_UP:
        lr = init_value
    elif epoch >= config.WARM_UP and epoch < 2*config.WARM_UP:
        # Calculate the exponential growth factor
        # Formula: lr = initial_value * (final_value / initial_value) ** (epoch / num)
        growth_factor = (final_value / init_value) ** ((epoch-config.WARM_UP) / num)
        lr = init_value * growth_factor
    else:
        lr = final_value


    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    
def correct_boxes(boxes):
    corrected_boxes = []
    for box in boxes:
        x, y, width, height, class_label = box
        # Calcular x_min y x_max a partir de las coordenadas del centro y la anchura
        x_min = x - width / 2
        x_max = x + width / 2
        y_min = y - height / 2
        y_max = y + height / 2

        # Asegurar que x_min y x_max están dentro del rango [0, 1]
        x_min = max(0, min(x_min, 1))
        x_max = max(0, min(x_max, 1))
        y_min = max(0, min(y_min, 1))
        y_max = max(0, min(y_max, 1))

        # Recalcular el centro y el tamaño basado en los límites ajustados
        new_width = x_max - x_min
        new_height = y_max - y_min
        new_x = x_min + new_width / 2
        new_y = y_min + new_height / 2

        corrected_boxes.append([new_x, new_y, new_width, new_height, class_label])
    return corrected_boxes 
# LABEL_DICT = {
#     1: "aeroplane",
#     2: "bicycle",
#     3: "bird",
#     4: "boat",
#     5: "bottle",
#     6: "bus",
#     7: "car",
#     8: "cat",
#     9: "chair",
#     10: "cow",
#     11: "dog",
#     12: "horse",
#     13: "motorbike",
#     14: "person",
#     15: "sheep",
#     16: "sofa",
#     17: "table",
#     18: "potted plant",
#     19: "train",
#     20: "tv/monitor"
# }

LABEL_DICT = {
    0: "aeroplane",
    1: "bicycle",
    2: "bird",
    3: "boat",
    4: "bottle",
    5: "bus",
    6: "car",
    7: "cat",
    8: "chair",
    9: "cow",
    10: "dining table",
    11: "dog",
    12: "horse",
    13: "motorbike",
    14: "person",
    15: "potted plant",
    16: "sheep",
    17: "sofa",
    18: "train",
    19: "tv/monitor"
}