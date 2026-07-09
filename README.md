# 🧠 FashionMNIST Classification using MLP with Dropout (PyTorch)

## 📌 Project Overview

This project implements a **Multi-Layer Perceptron (MLP)** with **Dropout Regularization** using **PyTorch** to classify images from the **FashionMNIST** dataset.

FashionMNIST is a dataset containing grayscale images of clothing items such as shirts, shoes, bags, and dresses. The model learns to classify each image into one of **10 categories**.

---

# 📂 Dataset

**Dataset:** FashionMNIST

- Training images: **60,000**
- Test images: **10,000** *(See note below)*
- Image size: **28 × 28 pixels**
- Color channels: **1 (Grayscale)**
- Classes: **10**

| Label | Class |
|--------|----------------|
| 0 | T-shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

# 📦 Import Libraries

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
```

These libraries are used for:

- Building neural networks
- Loading datasets
- Applying transformations
- Training models

---

# 🔄 Transform

```python
transform = transforms.ToTensor()
```

Converts images into PyTorch tensors.

It also normalizes pixel values:

```
0 - 255
```

into

```
0.0 - 1.0
```

making training faster and more stable.

---

# 📥 Loading Dataset

```python
train_data = datasets.FashionMNIST(
    root="/data",
    train=True,
    transform=transform,
    download=True
)
```

Loads the training dataset.

---

```python
test_data = datasets.FashionMNIST(
    root="/data",
    train=True,
    transform=transform,
    download=True
)
```

Loads the test dataset.

⚠ **Bug**

This should be:

```python
train=False
```

Correct version:

```python
test_data = datasets.FashionMNIST(
    root="/data",
    train=False,
    transform=transform,
    download=True
)
```

Otherwise the model is evaluated on the training data instead of the actual test data.

---

# 📦 DataLoader

```python
train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)
```

- Loads 64 images at a time
- Shuffles images every epoch
- Improves learning

---

```python
test_loader = DataLoader(
    test_data,
    batch_size=1000
)
```

Loads 1000 test images per batch.

No need to shuffle test data.

---

# 🧠 Model Architecture

```python
class MLPDropout(nn.Module):
```

Creates a custom neural network.

---

## Layer 1

```python
self.fc1 = nn.Linear(28*28,256)
```

Input:

```
784 pixels
```

Output:

```
256 neurons
```

---

## Dropout Layer

```python
self.drop1 = nn.Dropout(0.5)
```

Randomly disables **50% of neurons** during training.

Purpose:

- Prevent overfitting
- Improve generalization

---

## Layer 2

```python
self.fc2 = nn.Linear(256,128)
```

Transforms

```
256 → 128
```

features.

---

## Dropout

```python
self.drop2 = nn.Dropout(0.5)
```

Again randomly disables half of the neurons.

---

## Output Layer

```python
self.out = nn.Linear(128,10)
```

Produces

```
10 outputs
```

One score for each clothing category.

---

# 🔁 Forward Pass

## Flatten Image

```python
x = x.view(-1,28*28)
```

Converts

```
(64,1,28,28)
```

into

```
(64,784)
```

because Linear layers accept vectors, not images.

---

## First Hidden Layer

```python
x = F.relu(self.fc1(x))
```

Applies

```
Linear
↓

ReLU
```

ReLU removes negative values.

---

## Dropout

```python
x = self.drop1(x)
```

Randomly removes 50% of neurons.

---

## Second Hidden Layer

```python
x = F.relu(self.fc2(x))
```

Learns more complex image features.

---

## Dropout

```python
x = self.drop2(x)
```

Again disables random neurons.

---

## Output Layer

```python
x = self.out(x)
```

Returns raw prediction scores (**logits**).

No Softmax is used because `CrossEntropyLoss` applies it internally.

---

# 🏗 Create Model

```python
model = MLPDropout()
```

Creates the neural network.

---

# Loss Function

```python
criterion = nn.CrossEntropyLoss()
```

Measures how different predictions are from the correct labels.

Suitable for multi-class classification.

---

# Optimizer

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

Adam updates model weights using gradients.

Learning rate:

```
0.001
```

---

# 🚀 Training Loop

```python
for epoch in range(5):
```

Train for

```
5 epochs
```

---

## Forward Pass

```python
outputs = model(images)
```

Predicts class scores.

---

## Compute Loss

```python
loss = criterion(outputs, labels)
```

Calculates prediction error.

---

## Clear Old Gradients

```python
optimizer.zero_grad()
```

Removes gradients from the previous batch.

---

## Backpropagation

```python
loss.backward()
```

Computes gradients for every trainable parameter.

---

## Update Weights

```python
optimizer.step()
```

Updates weights using Adam.

---

## Print Loss

```python
print(f"the loss is {loss:.4f} in epoch {epoch+1}")
```

Displays the loss after each epoch.

Example:

```
the loss is 0.3125 in epoch 5
```

---

# 🧪 Model Evaluation

```python
correct = 0
total = 0
```

Counters for accuracy.

---

## Disable Gradient Calculation

```python
with torch.no_grad():
```

Improves speed and saves memory during testing.

---

## Predict Classes

```python
outputs = model(images)
```

Produces logits.

---

```python
_, predicted = torch.max(outputs,1)
```

Selects the class with the highest score.

Example:

```
Outputs

[1.2,0.3,5.1,0.8,...]

↓

Predicted class = 2
```

---

## Count Samples

```python
total += labels.size(0)
```

Adds the number of images in the current batch.

---

## Count Correct Predictions

```python
correct += (predicted == labels).sum().item()
```

Counts how many predictions match the true labels.

---

## Accuracy

```python
print(f"correctness is {(correct/total):.4f}")
```

Example output:

```
correctness is 0.8834
```

meaning

```
88.34% accuracy
```

---

# 📊 Network Structure

```
28×28 Image
      │
      ▼
Flatten (784)
      │
      ▼
Linear (784 → 256)
      │
      ▼
ReLU
      │
      ▼
Dropout (50%)
      │
      ▼
Linear (256 → 128)
      │
      ▼
ReLU
      │
      ▼
Dropout (50%)
      │
      ▼
Linear (128 → 10)
      │
      ▼
CrossEntropyLoss
```

---

# 🔧 Improvements

- Fix the test dataset by setting `train=False`.
- Normalize images using:

```python
transforms.Normalize((0.5,), (0.5,))
```

- Use GPU if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

- Save the trained model:

```python
torch.save(model.state_dict(), "fashion_mlp_dropout.pth")
```

- Call `model.eval()` before evaluation and `model.train()` before training to ensure Dropout behaves correctly.

---

# 📚 Concepts Covered

- PyTorch
- FashionMNIST Dataset
- DataLoader
- Tensor Transformations
- Multi-Layer Perceptron (MLP)
- Linear Layers
- ReLU Activation
- Dropout Regularization
- Forward Propagation
- Backpropagation
- CrossEntropyLoss
- Adam Optimizer
- Model Evaluation
- Classification Accuracy

---

# 🎯 Expected Accuracy

With the corrected test dataset (`train=False`) and this architecture, the model typically achieves **around 85–90% test accuracy** after a few epochs.
