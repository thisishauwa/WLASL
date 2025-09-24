import math
import os
import argparse

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms

import numpy as np

import torch.nn.functional as F
from pytorch_i3d import InceptionI3d

# from nslt_dataset_all import NSLT as Dataset
from datasets.nslt_dataset_all import NSLT as Dataset
import cv2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)

args = parser.parse_args()


def load_rgb_frames_from_video(video_path, start=0, num=-1):
    vidcap = cv2.VideoCapture(video_path)

    frames = []

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start)
    if num == -1:
        num = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    for offset in range(num):
        success, img = vidcap.read()
        if not success:
            break

        w, h, c = img.shape
        sc = 224 / w
        img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)

        img = (img / 255.) * 2 - 1

        frames.append(img)

    return torch.Tensor(np.asarray(frames, dtype=np.float32))


def run(init_lr=0.1,
        max_steps=64e3,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        batch_size=3 * 15,
        save_model='',
        weights=None,
        use_cuda=False):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=2,
                                                 pin_memory=False)

    dataloaders = {'test': val_dataloader}
    datasets = {'test': val_dataset}

    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('weights/flow_imagenet.pt', map_location=torch.device('cpu')))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt', map_location=torch.device('cpu')))
    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    
    if use_cuda:
        i3d.cuda()
        i3d = nn.DataParallel(i3d)
    
    i3d.eval()

    correct = 0
    correct_5 = 0
    correct_10 = 0

    top1_fp = np.zeros(num_classes, dtype=np.int)
    top1_tp = np.zeros(num_classes, dtype=np.int)

    top5_fp = np.zeros(num_classes, dtype=np.int)
    top5_tp = np.zeros(num_classes, dtype=np.int)

    top10_fp = np.zeros(num_classes, dtype=np.int)
    top10_tp = np.zeros(num_classes, dtype=np.int)

    for data in dataloaders["test"]:
        inputs, labels, video_id = data  # inputs: b, c, t, h, w
        
        if use_cuda:
            inputs = inputs.cuda()
            labels = labels.cuda()

        per_frame_logits = i3d(inputs)

        predictions = torch.max(per_frame_logits, dim=2)[0]
        out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
        out_probs = np.sort(predictions.cpu().detach().numpy()[0])

        if labels[0].item() in out_labels[-5:]:
            correct_5 += 1
            top5_tp[labels[0].item()] += 1
        else:
            top5_fp[labels[0].item()] += 1
        if labels[0].item() in out_labels[-10:]:
            correct_10 += 1
            top10_tp[labels[0].item()] += 1
        else:
            top10_fp[labels[0].item()] += 1
        if torch.argmax(predictions[0]).item() == labels[0].item():
            correct += 1
            top1_tp[labels[0].item()] += 1
        else:
            top1_fp[labels[0].item()] += 1
        print(video_id, float(correct) / len(dataloaders["test"]), float(correct_5) / len(dataloaders["test"]),
              float(correct_10) / len(dataloaders["test"]))

        # per-class accuracy
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp))
    print('top-k average per class acc: {}, {}, {}'.format(top1_per_class, top5_per_class, top10_per_class))


def run_on_tensor(weights, ip_tensor, num_classes, use_cuda=False):
    i3d = InceptionI3d(400, in_channels=3)

    i3d.replace_logits(num_classes)
    i3d.load_state_dict(torch.load(weights, map_location=torch.device('cpu')))
    
    if use_cuda:
        i3d.cuda()
        i3d = nn.DataParallel(i3d)
        ip_tensor = ip_tensor.cuda()
    
    i3d.eval()

    t = ip_tensor.shape[2]
    per_frame_logits = i3d(ip_tensor)

    predictions = F.upsample(per_frame_logits, t, mode='linear')

    predictions = predictions.transpose(2, 1)
    out_labels = np.argsort(predictions.cpu().detach().numpy()[0])

    arr = predictions.cpu().detach().numpy()[0,:,0].T

    plt.plot(range(len(arr)), F.softmax(torch.from_numpy(arr), dim=0).numpy())
    plt.show()

    return out_labels


def run_on_single_video(video_path, weights, num_classes, use_cuda=False):
    """Run inference on a single video file."""
    print(f"Running inference on {video_path}")
    
    # Load and preprocess video
    frames = load_rgb_frames_from_video(video_path)
    frames = frames.unsqueeze(0).permute(0, 4, 1, 2, 3)  # [B, C, T, H, W]
    
    # Run inference
    out_labels = run_on_tensor(weights, frames, num_classes, use_cuda)
    
    # Get top predictions
    top_indices = out_labels[-10:][::-1]  # Top 10 predictions in descending order
    
    # Load class names
    try:
        with open('preprocess/wlasl_class_list.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        
        print("\nTop 10 predictions:")
        for i, idx in enumerate(top_indices):
            if idx < len(class_names):
                print(f"{i+1}. {class_names[idx]}")
            else:
                print(f"{i+1}. Class {idx}")
    except:
        print("\nTop 10 predictions (class indices):", top_indices)
    
    return top_indices


if __name__ == '__main__':
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    print(f"CUDA available: {use_cuda}")
    
    # ================== test i3d on a dataset ==============
    mode = 'rgb'
    num_classes = 2000
    save_model = './checkpoints/'

    root = '../../data/WLASL2000'

    train_split = 'preprocess/nslt_{}.json'.format(num_classes)
    weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'

    # Check if videos are available for testing
    import os
    test_videos = []
    if os.path.exists(root):
        test_videos = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.mp4')]
    
    if test_videos:
        print(f"Found {len(test_videos)} test videos. Running inference on the first video...")
        run_on_single_video(test_videos[0], weights, num_classes, use_cuda)
    else:
        print("No test videos found. Running on dataset...")
        run(mode=mode, root=root, save_model=save_model, train_split=train_split, weights=weights, use_cuda=use_cuda)
