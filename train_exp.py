import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import CNN
# Hyperparameters
EPOCH = 5
LR = 0.001
BATCH_SIZE = 50

# Load Data
class DigitDataset(Dataset):
    def __init__(self,root):
        self.root = root
        self.files = os.listdir(root)
        self.transform = transforms.ToTensor()   # 会把(H,W)转成(1,H,W)
    def __len__(self):
        return len(self.files)
    def __getitem__(self,idx):
        fname = self.files[idx]
        label = int(fname[0])
        path = os.path.join(self.root,fname)

        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE) 
        img = self.transform(img)  # 转成tensor，自动 /255
        return img,label
# Create dataset and dataloader
train_data = DigitDataset('./TrainingSet/')
test_data = DigitDataset('./TestSet/')

train_loader = Data.DataLoader(train_data, batch_size=50, shuffle=True)
test_loader = Data.DataLoader(test_data, batch_size=200, shuffle=False)
# Initialize CNN
cnn = CNN.CNN()
print(cnn)
# Optimizer and loss function
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if step % 100 == 0:
            # 测试集计算 accuracy
            correct = 0
            total = 0
            for test_x, test_y in Data.DataLoader(test_data, batch_size=200, shuffle=False):
                test_output = cnn(test_x)
                pred = torch.max(test_output, 1)[1]
                correct += (pred == test_y).sum().item()
                total += test_y.size(0)

            acc = correct / total
            print(f'Epoch {epoch} | Step {step} | Loss {loss.item():.4f} | Test Acc {acc:.4f}')

# Final evaluation on test set

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(loss_list)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.plot(acc_list)
plt.title("Test Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

all_preds = []
all_labels = []
for t_x, t_y in test_loader:
    out = cnn(t_x)
    pred = torch.max(out, 1)[1]
    all_preds.extend(pred.numpy())
    all_labels.extend(t_y.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

print("Classification Report:")
print(classification_report(all_labels, all_preds))

wrong_imgs = []
wrong_pred = []
wrong_true = []

for img, label in test_data:
    output = cnn(img.unsqueeze(0))
    pred = torch.max(output,1)[1].item()
    if pred != label:
        wrong_imgs.append(img.squeeze().numpy())
        wrong_pred.append(pred)
        wrong_true.append(label)
    if len(wrong_imgs) >= 16:
        break

plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(wrong_imgs[i], cmap='gray')
    plt.title(f"T:{wrong_true[i]} P:{wrong_pred[i]}")
    plt.axis('off')

plt.suptitle("Failure Cases (Top 16)")
plt.show()

torch.save(cnn.state_dict(), 'digit_cnn.pkl')
print("all is ok. And Model saved as digit_cnn.pkl")