import torch
import os
import pandas as pd
from PIL import Image
import numpy as np
from utils import extract_image_name, correct_boxes
from torchvision import transforms as T

class YOLOdataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, label_dir, S=4, B=2, C=3, transform=None, mode = "train"):
        #print(csv_file)
        self.img_dir = img_dir+f'/{mode}' #os.path.join(img_dir, mode)
        self.label_dir = label_dir+f'/{mode}'#os.path.join(label_dir, mode)
        image_name, label_name = extract_image_name(self.img_dir)
        self.annotations = pd.DataFrame({'img': image_name, 'label': label_name})         
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                if label != "\n":
                    class_label, x, y, width, height = [
                        float(x) if float(x) != int(float(x)) else int(float(x))
                        for x in label.replace("\n", "").split()
                    ]
                    
                    boxes.append([x, y, width, height, class_label])
        boxes = correct_boxes(boxes)
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        # image = np.array(Image.open(img_path).convert("RGB"))
        image = Image.open(img_path).convert("RGB")
        # print(label_path)
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # image = T.ToTensor()(image) if not isinstance(image, torch.FloatTensor) else image.float()
        # Convert To Cells
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B)) # (4, 4, 13)
        for box in boxes:
            # box is relative to the image size
            x, y, width, height, class_label = box
            class_label = int(class_label)

            # i,j represents the cell row and cell column
            i, j = int(self.S * y), int(self.S * x) # row, column
            # x_cell, y_cell represents the position of the box in the cell
            # they are relative to the cell size, whereas x,y relative to the entire image
            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """            
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S, # width of the box relative to the entire image
                height * self.S
            ) # width_cell, height_cell are relative to the cell size
            
            if label_matrix[i, j, -10] == 0:
                # Set the box
                label_matrix[i, j, -10] = 1
                # Box coordinates
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, -9:-5] = box_coordinates
                # Set one hot encoding for class_label
                
                label_matrix[i, j, class_label] = 1
        print(label_matrix.shape)
        return image, label_matrix

    
def test():
    import config
    from torch.utils.data import DataLoader
    from utils import plot_image, cellboxes_to_boxes, non_max_suppression, cellboxes_to_boxes_test
    from tqdm import tqdm
    transform = config.test_transform

    dataset = YOLOdataset(
        img_dir='D:\\Usuarios\\Carlos\\Documentos\\MUIT\\TFM\\Code\\AirbusAircraft\\archive\\data_YOLO_256\\images',
        label_dir="D:\\Usuarios\\Carlos\\Documentos\\MUIT\\TFM\\Code\\AirbusAircraft\\archive\\data_YOLO_256\\labels",
        S=100,
        B=2,
        transform=transform,
        mode = "test",
        C=config.NUM_CLASSES,
    )


    loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    for x, y in loader:
        boxes = []
        print(y[0].shape)
        # print(y)
        for i in range(8):
            bboxes = cellboxes_to_boxes_test(y, SPLIT_SIZE=100, NUM_BOXES=2, NUM_CLASSES=config.NUM_CLASSES)
        
            boxes = non_max_suppression(bboxes[i], iou_threshold=0.5, threshold=0.4, box_format="midpoint")            
            plot_image(x[i].permute(1, 2, 0).to("cpu"), boxes)
        # print(x[0])
       

        


if __name__ == "__main__":
    test()



    
