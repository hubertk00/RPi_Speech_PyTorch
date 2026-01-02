import torch

model_path = r"C:\Users\Hubert\Desktop\Praca_dyplomowa_PyTorch\wakeword_models\resnet14_wakeword.pth"

checkpoint = torch.load(model_path, map_location='cpu')

accuracy = checkpoint['accuracy']
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"Dokładność (Validation Acc): {accuracy * 100:.2f}%")
print(f"Epoka: {epoch}")
print(f"Loss: {loss:.4f}")

