import torch
from torchvision import transforms
from PIL import Image, ImageOps
from models import get_model
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from models.cnn import TrafficSignModel
import numpy as np
import os
import warnings
from ultralytics import YOLO
import random
import string 
import io

def save_image(image_bytes, prefix):
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    filename = f"{prefix}_{random_suffix}.jpg"
    filepath = os.path.join('localized_obj', filename)
    with open(filepath, 'wb') as f:
        f.write(image_bytes.getbuffer())
    return filepath

def pad_to_square(image):
    width, height = image.size

    max_side = max(width, height)

    # Calculate padding
    left = (max_side - width) // 2
    right = max_side - width - left
    top = (max_side - height) // 2
    bottom = max_side - height - top

    # Pad the image
    padded_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=(255,255,255))

    return padded_image

def load_image(image_path, img_cols, img_rows):
    transform = transforms.Compose([
        transforms.Resize((img_rows, img_cols)),
        transforms.ToTensor()
    ])
    image = pad_to_square(image)
    
    binn = io.BytesIO()
    image.save(binn, format='JPEG')
    binn.seek(0)
    path_crop = save_image(binn, 'crop')
    
    image = transform(image).unsqueeze(0)
    return image, path_crop

def load_model(extractor, classifier, localizer_path):
    
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    net1 = get_model('sillnet', 43, 43, 6)
    net1.to(device='cuda')
    net1 = torch.load(extractor,map_location=torch.device('cuda'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net2 = TrafficSignModel().to(device)
    net2.load_state_dict(torch.load(classifier, map_location=device))
    net2.eval()

    stn_weight = YOLO(localizer_path)

    return (net1, net2, stn_weight)

def localizer(original_image, stn_weight):
    new_width = 64
    aspect_ratio = original_image.height / original_image.width
    new_height = int(new_width * aspect_ratio)
    resized_image = original_image.resize((new_width, new_height))
    blank_image = Image.new("RGB", (640, 640), color=(255, 255, 255))
    x_offset = (640 - new_width) // 2  # Horizontal offset
    y_offset = (640 - new_height) // 2  # Vertical offset
    blank_image.paste(resized_image, (x_offset, y_offset))
    
    binn = io.BytesIO()
    blank_image.save(binn, format='JPEG')
    binn.seek(0)
    obj_path = save_image(binn, 'localize')
    results = stn_weight(obj_path)
    
    threshold = 0.4
    x1, y1, x2, y2 = (0,0,0,0)
    for result in results:
        for box in result.boxes:
            bbox = box.xyxy[0].tolist()  # Get the bounding box [x1, y1, x2, y2]
            confidence = box.conf[0].item()  # Confidence score
            #class_id = int(box.cls[0].item())  # Class ID
            #class_name = result.names[class_id]  # Class name
            if confidence > threshold:
                threshold = confidence
                x1, y1, x2, y2 = bbox
    if threshold > 0.4: 
        image_cropped = Image.open(static_path).convert('RGB')
        image_cropped = image_cropped.crop((x1, y1, x2, y2)) 
        return image_cropped
    
    return original_image

def inference(net, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net1, net2, stn_weight = net

    image = Image.open(image_path).convert('RGB')

    image_localized = localizer(image, stn_weight)

    image, path_crop = load_image(image_localized, 64, 64).to(device)
    with torch.no_grad():
        feat_sem, feat_illu, _ = net1.extract(image, is_warping=True)
        
        recon_feat_sem = net1.decode(feat_sem)
        recon_feat_illu = net1.decode(feat_illu)

    # Convert images to bytes
    buffer_sem = io.BytesIO()
    buffer_illu = io.BytesIO()
    torchvision.utils.save_image(recon_feat_sem, buffer_sem, format='JPEG')
    torchvision.utils.save_image(recon_feat_illu, buffer_illu, format='JPEG')
    buffer_sem.seek(0)
    buffer_illu.seek(0)

    # Continue with net2 inference
    image = Image.open(path_crop)
    image = image.resize((30, 30))
    image = np.array(image)
    image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = net2(image_tensor)
        _, pred_class = torch.max(outputs, 1)

    return buffer_sem, buffer_illu, pred_class.item()

    
