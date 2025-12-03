import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm 
import os
import random
from neuralnet.dataset import Dataset
from neuralnet.MatchboxNet import MatchboxNet
from neuralnet.utils import load_wake_word

class Config:
    DATA_ROOT = r"C:\Users\Hubert\Desktop\Praca_dyplomowa_PyTorch\Nagrania"
    NOISE_PATHS = [
        r"background/doing_the_dishes.wav",
        r"background/running_tap.wav",
        r"background/dude_miaowing.wav",
        r"background/excercise_bike.wav"
    ]
    WAKE_WORD = "Hugo"       
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    SAMPLE_RATE = 16000
    NUM_WORKERS = 4
    SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, loader, criterion, optimizer, device, epoch_idx, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    loop = tqdm(loader, desc=f"Epoch {epoch_idx}/{total_epochs} [Train]")
    
    for inputs, targets in loop:
        inputs = inputs.to(device)
        targets = targets.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().unsqueeze(1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    avg_loss = running_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    #set_seed(Config.SEED)
    device = get_device()

    file_paths, labels = load_wake_word(Config.DATA_ROOT, wake_word=Config.WAKE_WORD)
    
    num_pos = sum(labels)
    num_neg = len(labels) - num_pos

    X_train, X_val, y_train, y_val = train_test_split(
        file_paths, labels, test_size=0.3, stratify=labels)

    train_dataset = Dataset(X_train, y_train, sample_rate=Config.SAMPLE_RATE, augment=True, noise_paths=Config.NOISE_PATHS)
    test_dataset = Dataset(X_val, y_val, sample_rate=Config.SAMPLE_RATE, augment=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=Config.NUM_WORKERS, 
        pin_memory=True, 
        persistent_workers=True
    )

    model = MatchboxNet(input_channels=40, num_classes=1, B=3, R=1, C=64).to(device)
    pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, Config.EPOCHS)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            print(f" >> Zapisywanie modelu ({best_val_loss:.4f} -> {val_loss:.4f})")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_wakeword.pth")

if __name__ == '__main__':
    main()