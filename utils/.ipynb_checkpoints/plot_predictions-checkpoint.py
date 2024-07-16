from PIL import Image, ImageDraw, ImageFont
import os 
import sys
# Custom imports
sys.path.append(os.getcwd())
import config


def plot_image_with_pil_(image, pred_boxes, true_boxes=None, colors = None):
    """Edit the image to draw bounding boxes using PIL but does not show or save the image."""
    class_labels = config.LABELS

    # Define colores utilizando PIL
    num_classes = len(class_labels)
    if colors is None:
        colors = config.colors[:num_classes]
    else:
        colors = colors
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for box in pred_boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        confidence_score = box[1]
        x, y, w, h = box[2], box[3], box[4], box[5]

        upper_left_x = (x - w / 2) * width
        upper_left_y = (y - h / 2) * height
        lower_right_x = (x + w / 2) * width
        lower_right_y = (y + h / 2) * height
        draw.rectangle([upper_left_x, upper_left_y, lower_right_x, lower_right_y], outline=colors[class_pred], width=3)
        draw.text((upper_left_x, upper_left_y), f'{confidence_score:.2f} {class_labels[class_pred]}', fill=colors[class_pred], font=ImageFont.truetype(os.path.join(os.getcwd(),"utils/font_text/04B_08__.TTF"), 20))
    
    if true_boxes is not None:
        for box in true_boxes:
            if box[1] > 0.4:
                assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
                class_pred = int(box[0])
                confidence_score = box[1]
                x, y, w, h = box[2], box[3], box[4], box[5]

                upper_left_x = (x - w / 2) * width
                upper_left_y = (y - h / 2) * height
                lower_right_x = (x + w / 2) * width
                lower_right_y = (y + h / 2) * height
                draw.rectangle([upper_left_x, upper_left_y, lower_right_x, lower_right_y], outline='red', width=5)
    return image