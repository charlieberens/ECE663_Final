import argparse
import os
import numpy as np
from resnet import resnet50
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
import torch
import matplotlib.pyplot as plt

def get_model(weights_path):
    model = resnet50(num_classes=1)
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    return model

def fgsm(model, x, y, eps):
    x.requires_grad = True
    model.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(x).sigmoid().flatten(), y.float())
    loss.backward()
    x_adv = x + eps * x.grad.sign()
    return x_adv

def get_classes(dataset_path):
    return [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]

def get_dataset(dataset_path, eps):
    print("Generating adversarial dataset")
    dset_lst = []
    for cls in get_classes(dataset_path):
        print(f"Generating adversarial examples for class {cls}")
        root = os.path.join(dataset_path, cls)
        dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
        dset_lst.append(dset)
        
    return torch.utils.data.ConcatDataset(dset_lst)

def get_dataloader(dataset_path, method, eps):
    dset = get_dataset(dataset_path, eps)
    loader = torch.utils.data.DataLoader(dset, batch_size=64, shuffle=True, num_workers=2)
    return loader

def evaluate_model(model, dataloader, eps):
    print("Evaluating model")
    # with torch.no_grad():
    y_true, y_pred = [], []
    num_iter = 0
    for img, label in dataloader:
        num_iter += 1
        in_tens = img.cuda()
        label = label.cuda()
        adversarial_img = fgsm(model, in_tens, label, eps)
        y_pred.extend(model(adversarial_img).sigmoid().flatten().tolist())
        y_true.extend(label.flatten().tolist())

        if num_iter % 1 == 0:
            y_true_2, y_pred_2 = np.array(y_true), np.array(y_pred)
            r_acc = accuracy_score(y_true_2[y_true_2==0], y_pred_2[y_true_2==0] > 0.5)
            f_acc = accuracy_score(y_true_2[y_true_2==1], y_pred_2[y_true_2==1] > 0.5)
            acc = accuracy_score(y_true_2, y_pred_2 > 0.5)
            ap = average_precision_score(y_true_2, y_pred_2)

            print(f"\nAccuracy: {acc}")
            print(f"Real Accuracy: {r_acc}")
            print(f"Fake Accuracy: {f_acc}")
            print(f"Average Precision: {ap}")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    print(f"Accuracy: {acc}")
    print(f"Real Accuracy: {r_acc}")
    print(f"Fake Accuracy: {f_acc}")
    print(f"Average Precision: {ap}")

def show_adversarial_examples(model, dataloader, eps):
    adv_images = []
    images = []
    labels = []
    for img, label in dataloader:
        # Append the first 4 images in the batch
        adversarial_img = fgsm(model, img.cuda(), label.cuda(), eps)
        adv_images.extend(adversarial_img[:4].cpu().detach().numpy())
        images.extend(img[:4].numpy())
        labels.extend(label[:4].numpy())

        print(img[0])
        print(adv_images[0])

        break

    # Unnormalize the images
    images = [np.clip((i * 0.229 + 0.485), 0, 1) for i in images]
    adv_images = [np.clip((i * 0.229 + 0.485), 0, 1) for i in adv_images]

    # Create a grid of images
    fig, ax = plt.subplots(2, 4, figsize=(20, 20))
    for i in range(4):
        ax[0, i].imshow(images[i].transpose(1, 2, 0))
        ax[0, i].set_title(f"Real: {labels[i]}")
        ax[0, i].axis('off')

        ax[1, i].imshow(adv_images[i].transpose(1, 2, 0))
        ax[1, i].set_title(f"Adversarial: {labels[i]}")
        ax[1, i].axis('off')

    fig.tight_layout()
    
    # Save the plot
    plt.savefig("adversarial_examples.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluate model with adversarial examples')
    parser.add_argument('-d', type=str, help='Path to dataset folder. The folder should contain 0_real, and 1_fake folders')
    parser.add_argument('-m', type=str, help='Path to model weights')
    parser.add_argument("--method", type=str, help="Adversarial attack method")
    parser.add_argument("--eps", type=float, help="Epsilon value for adversarial attack")

    args = parser.parse_args()

    if args.method == "fgsm":
        print("Evaluating model with FGSM adversarial examples")
    elif args.method == "deepfool":
        print("Evaluating model with DeepFool adversarial examples")
    else:
        print("Invalid adversarial attack method")

    print("Dataset folder: ", args.d)
    print("Model weights: ", args.m)
    print("Epsilon value: ", args.eps)

    model = get_model(args.m)
    dataloader = get_dataloader(args.d, args.method, args.eps)
    show_adversarial_examples(model, dataloader, args.eps)
    evaluate_model(model, dataloader, args.eps)

if __name__ == '__main__':
    main()