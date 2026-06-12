
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import numpy as np
from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models

class AugmentedObjectsDataset(Dataset):
    """
    Dataset class for custom object detection data.
    Supports PASCAL VOC format and COCO format.
    """
    def __init__(self, root_dir="./", albumentation_transforms=None, train=True):
        """
        Args:
            root_dir: Directory with all the images and annotations
            transforms: Albumentations transforms
            train: Whether this is training data
        """
        self.root_dir = Path(root_dir)
        self.albumentation_transforms = albumentation_transforms
        self.train = train
        
        # Define class mapping - always include background as class 0
        # self.arr_class_names = ['background']  # Background always index 0
        self.class_to_idx = {'background': 0}
        for i in range(0, len(str_classes)):
            self.class_to_idx[str_classes[i]] = i
        
        # Load images and annotations
        self.images = []
        self.annotations = []
        
        self._load_annotation_data()
        
        print(f"Loaded {len(self.images)} images with {sum(len(anns) for anns in self.annotations)} total annotations")
        print(f"Classes: {str_classes}")
    
    def _load_annotation_data(self):
        # Find all images
        image_dir = self.root_dir / "data/validation"

        if self.train:
            image_dir = self.root_dir / "data/training"

        # ann_dir = self.root_dir / "Annotations"
        
        if not image_dir.exists():
            # Alternative: images are in root directory
            image_files = list(self.root_dir.glob("*.jpg")) + \
                         list(self.root_dir.glob("*.jpeg")) + \
                         list(self.root_dir.glob("*.png"))
            # ann_files = list(self.root_dir.glob("*.xml"))
        else:
            image_files = list(image_dir.glob("*.jpg")) + \
                         list(image_dir.glob("*.jpeg")) + \
                         list(image_dir.glob("*.png"))
            # ann_files = list(ann_dir.glob("*.xml"))
        
        # Match images with annotations
        for img_path in image_files:
            # ann_path = None
            ann_path = img_path.parent / f"{img_path.stem}.txt"
            # if ann_dir.exists():
            #     ann_path = ann_dir / f"{img_path.stem}.xml"
            # else:
            #     # Look for XML in same directory
            #     ann_path = img_path.parent / f"{img_path.stem}.xml"
            
            if ann_path.exists():
                # boxes, labels, class_names = self._parse_voc_xml(ann_path)
                # parse label in the image from txt to number
                class_name = 'background'
                try:
                    class_name = ann_path.read_text().strip()
                except Exception as e:
                    print(e)

                # if class_name not in self.class_to_idx:
                #     self.class_to_idx[class_name] = len(self.arr_class_names)
                #     self.arr_class_names.append(class_name)

                self.images.append(str(img_path))
                self.annotations.append({
                    'label': self.class_to_idx[class_name],
                    # 'class_names': class_names
                })
                
                # if boxes:  # Only add if there are annotations
                #     self.images.append(str(img_path))
                #     self.annotations.append({
                #         'boxes': boxes,
                #         'labels': labels,
                #         # 'class_names': class_names
                #     })
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann = self.annotations[idx]
        # label = np.array(ann['label'], dtype=np.int64)
        label = ann['label']

        # print('.debug....')
        # print(self.images)
        # print(self.annotations)
        
        # Apply transforms
        if self.albumentation_transforms:
            image = self.albumentation_transforms(
                image=image,
            )['image']
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((122, 122)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #    std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)
        
        # Convert to tensors
        target = {}
        target['label'] = torch.as_tensor(label, dtype=torch.int64)
        
        return image, target
            
str_classes = ['rectangle', 'circle', 'triangle']

def fn_create_synthetic_data():
    """Create synthetic training data for demonstration"""
    import numpy as np
    from PIL import Image, ImageDraw
    
    data_dir = Path("data")
    train_dir = (data_dir / "training")
    val_dir = (data_dir / "validation")
    
    # Create directories
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    import os

    def fn_create_path_images(img_path):

        if os.listdir(img_path):
            # assume already created images will be reused
            print('assume already created images will be reused in ' + str(img_path))
            return

        # Create 400 training images
        for i in range(400):
            # Create blank image
            img = Image.new('RGB', (640, 480), color='white')
            draw = ImageDraw.Draw(img)
            
            # Create random shapes
            # num_shapes = np.random.randint(1, 4)
            annotations = []
            
            # for _ in range(num_shapes):
            shape_type = np.random.choice(str_classes)
            
            if shape_type == 'rectangle':
                x1, y1 = np.random.randint(100, 300), np.random.randint(100, 200)
                x2, y2 = x1 + np.random.randint(50, 300), y1 + np.random.randint(50, 200)
                draw.rectangle([x1, y1, x2, y2], outline='red', width=np.random.randint(10, 40))
            
            elif shape_type == 'circle':
                cx, cy = np.random.randint(150, 340), np.random.randint(150, 240)
                radius = np.random.randint(30, 120)
                draw.ellipse([cx-radius, cy-radius, cx+radius, cy+radius], 
                            outline='blue', width=np.random.randint(10, 40))
            
            elif shape_type == 'triangle':
                x1, y1 = np.random.randint(100, 300), np.random.randint(100, 200)
                size = np.random.randint(40, 200)
                points = [(x1, y1), (x1+size, y1), (x1+size//2, y1-size)]
                draw.polygon(points, outline='green', width=np.random.randint(10, 40))
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]

            annotations.append({
                'class': shape_type,
            })
            
            # Save image
            temp_path = img_path / f"train_{i:03d}.jpg"
            img.save(temp_path)
            
            # Save annotation (simple format)
            temp_path = img_path / f"train_{i:03d}.txt"
            with open(temp_path, 'w') as f:
                # for ann in annotations:
                ann = annotations[0]
                # class_idx = classes.index(ann['class'])
                # bbox = ann['bbox']
                # f.write(f"{class_idx}\n")
                f.write(f"{ann['class']}\n")
        
        print(f"  Created {len(list(img_path.glob('*.jpg')))} images")
        print(f"  Classes: {str_classes}")

    fn_create_path_images(train_dir)

    fn_create_path_images(val_dir)
    
    # fn_create_validation_images()

import albumentations as A

def fn_get_dataset(train=True):
    dataset = AugmentedObjectsDataset(
        train=train,
        albumentation_transforms=
                A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.RandomGamma(p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.ToGray(p=0.1),
                # A.Resize(height=800, width=800, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # ToTensorV2(),
            ],) 
                        #   bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))
                )
    return dataset

def collate_fn(batch):
    images = [item[0] for item in batch] # return image
    labels = [item[1]['label'] for item in batch] # return label
    return images, labels

def fn_get_dataloader(train=True):
    dataloader = DataLoader(
        dataset=fn_get_dataset(train=train),
        batch_size=40,
        shuffle=train,
        num_workers=2,
        collate_fn=collate_fn
    )
    return dataloader

# for batch_idx, (images, labels) in enumerate(dataloader):
#     print(batch_idx, (len(images), labels))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.str_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.model_features = vgg16(pretrained=True).features # shd be pretrain + freezed
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(222784, 512)
        self.fc2 = nn.Linear(512, len(str_classes))

        def model_torch_flatten(x):
            return torch.flatten(x, 1) # do not flatten first dimension of numbere of images
        self.model_torch_flatten = model_torch_flatten

        self.cls_criterion = nn.NLLLoss()

        self.optimizer = optim.Adadelta(self.parameters(), lr=1)
    
    def set_eval_mode(self):
        self.eval()
    
    def set_train_mode(self):
        self.train()
    
    def forward(self, images, train=True):
        if train:
            self.set_train_mode()
        else:
            self.set_eval_mode()

        self.optimizer.zero_grad()
        
        x = self.conv1(images)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output

    
    def train_epochs(
            self,
            train_data_loader,
            validation_data_loader,
            num_epoch=10,
            str_save_dir='checkpoints'
        ):

        history = {
            'train_cls_loss': [],
            'validation_cls_loss': [],
            'validation_accuracy': []
        }

        num_highest_accuracy = 0
        best_model = None

        for num_epoch in range(1, num_epoch + 1):
            """
            per epoch training
            """
            print(f"\n{'='*60}")
            print(f"Epoch {num_epoch}")
            print(f"{'='*60}")

            num_train_total_sample_size = 0
            num_train_cls_error = 0

            for _, (images, labels) in enumerate(train_data_loader):

                print('training batch....')
                images = torch.stack(images).to(self.str_device)
                labels = torch.stack(labels).to(self.str_device)
                # print(images.size())
                
                self.optimizer.zero_grad()
                # num_train_cls_prob = self.forward(images, train=True) 
                num_train_cls_prob = self(images) 
                # print(num_train_cls_prob)

                # loss = self.cls_criterion(
                #     num_train_cls_prob,
                #     labels
                # )

                loss = F.nll_loss(
                    num_train_cls_prob,
                    labels,
                )

                loss.backward()
                self.optimizer.step()

                # print(loss.item())

                num_train_cls_error += loss.item() * len(images)
                num_train_total_sample_size += len(images)

            num_train_cls_error /= num_train_total_sample_size

            history['train_cls_loss'].append(num_train_cls_error)

            """
            per epoch validation
            """

            num_validation_total_sample_size = 0
            num_validation_cls_error = 0
            num_total_correct = 0

            for _, (images, labels) in enumerate(validation_data_loader):

                images = torch.stack(images).to(self.str_device)
                labels = torch.stack(labels).to(self.str_device)

                with torch.no_grad():
                    num_validation_cls_prob = self.forward(images, train=False)
                    num_validation_cls_error += self.cls_criterion(
                        num_validation_cls_prob,
                        labels
                    ).item() * len(images)
                num_validation_total_sample_size += len(images)

                _, predicted = torch.max(num_validation_cls_prob, 1)
                correct = (predicted == labels).sum().item()

                num_total_correct += correct

            num_validation_cls_error /= num_validation_total_sample_size
            num_validation_accuracy = num_total_correct/num_validation_total_sample_size

            history['validation_cls_loss'].append(num_validation_cls_error)
            history['validation_accuracy'].append(num_validation_accuracy)

            if num_highest_accuracy <=  num_validation_accuracy:
                num_highest_accuracy = num_validation_accuracy
                torch.save(self.state_dict(), "best_model.pt")

            print(history)

    def validate(self, images, labels):
        with torch.no_grad():

            cls_scores = self.forward(images, train=False),

            if not labels:
                labels = torch.tensor([0]*images.size(0))

            # get labels
            cls_loss = self.cls_criterion(
                cls_scores,
                labels
            )

            _, predicted = torch.max(cls_scores, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct/labels.size(0)

            return cls_loss, accuracy
    
    ## [o] predict: 3.27.10-3
    """
    - chking input format
    """

    def predict(self, image_path):
        # load best model
        try:
            checkpoint = torch.load("best_model.pt")
            self.load_state_dict(checkpoint)
        except:
            pass

        # path to image (h, w, c)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # convert from (h,w,c) to (c,h,w)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((122, 122)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            #    std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)

        images = torch.stack([image])

        with torch.no_grad():
            if torch.cuda.is_available():
                image = image.cuda()
            # input: (c, h, w)
            cls_score = self.forward(images, train=False)
            # cls_probs = F.softmax(cls_score, dim=1)
            cls_probs = F.softmax(cls_score)

        for i in range(0, cls_probs.size(1)):
            print("probability of " + str_classes[i] + ": " + str(cls_probs[0][i].item()) + ".")
        # max_scores, pred_labels = torch.max(cls_probs, dim=1)
        # max_scores = max_scores.cpu().numpy()
        # pred_labels = pred_labels.cpu().numpy()

fn_create_synthetic_data()

# exit()

model = Model()

model.to(model.str_device)

# model.train_epochs(
#     fn_get_dataloader(train=True),
#     fn_get_dataloader(train=False),
#     num_epoch=10,
# )

model.predict(Path() / 'data/validation/train_006.jpg')