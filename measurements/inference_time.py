import torch
import time
import statistics
from neuralnet.MatchboxNet import MatchboxNet
from neuralnet.Resnet import ResNet8, ResNet14
from neuralnet.CRNN import CRNN
import argparse

INPUT_CHANNELS = 20  
INPUT_TIME = 101     
SAMPLE_INPUT = torch.randn(1, INPUT_CHANNELS, INPUT_TIME)
num_classes = 9


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, required=True, choices=['matchboxnet', 'resnet8', 'resnet14', 'crnn'])
    args = parser.parse_args()
    input_data = torch.randn(1, INPUT_CHANNELS, INPUT_TIME)
    if args.arch == 'matchboxnet':
        model = MatchboxNet(input_channels=INPUT_CHANNELS, num_classes=num_classes, B=3, R=1, C=64)
    elif args.arch == 'resnet8':
        model = ResNet8(input_channels=INPUT_CHANNELS, num_classes=num_classes, k=1.5)
    elif args.arch == 'resnet14':
        model = ResNet14(input_channels=INPUT_CHANNELS, num_classes=num_classes, k=1.5) 
    elif args.arch == 'crnn':
        model = CRNN(input_channels=INPUT_CHANNELS, num_classes=num_classes, k=1.5)

    model.eval()
    
    for i in range(10):
        i = model(input_data)

    num_iterations = 1000
    results = []

    for i in range(num_iterations):
        start_time = time.time()
        i = model(input_data)
        end_time = time.time()
        results.append(end_time - start_time)

    average = statistics.mean(results)*1000
    print(f"Sredni czas inferencji: {average:.5f}")