from multiprocessing import cpu_count, set_start_method, freeze_support
import torch
from torch import nn, optim
from torch.nn import functional as F
from typing import Callable
import torchvision.datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
import time

# # Configure logging
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def worker_init_fn(worker_id):
    logging.debug(f'Worker {worker_id} initializing')


# Model definition
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size) 
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)  
        x = self.l2(x) 
        return x

def statistical_parity(predictions, protected_attribute):
    # Dummy implementation: assumes protected_attribute is a tensor of 0s and 1s indicating two groups
    # Ensure predictions and protected_attribute are of floating point type for mean calculation
    predictions = predictions.float()
    protected_attribute = protected_attribute.float()
    group_0 = predictions[protected_attribute == 0]
    group_1 = predictions[protected_attribute == 1]
    parity = torch.abs(group_0.mean() - group_1.mean())
    return parity.item()

if __name__ == "__main__":
    set_start_method('spawn', force=True) # spawn? why here
    freeze_support()
    
    # Parameters
    batch_size = 8
    transform = transforms.ToTensor()
    num_workers = 4
    
    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    
    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=batch_size, pin_memory=True, num_workers=num_workers)

    # Instantiate the model
    feature_count = 3072
    hidden_layer_size = 100
    class_count = 10
    model = MLP(feature_count, hidden_layer_size, class_count)
    
    optimiser = optim.SGD(model.parameters(), lr=1e-6)
    loss_function = nn.CrossEntropyLoss()
    
    # Timing the training process
    start_time = time.time()
    
    try:
        for i in range(10):
            print(f"\nBatch {i+1} out of 10")
            
            LargestPar = 0
            
            for batch, labels in train_loader:
                batch = batch.flatten(1)
                logits = model(batch)
                loss = loss_function(logits, labels)
                
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()
                
                # Statistical parity fairness check (dummy example)
                protected_attribute = torch.randint(0, 2, (batch.size(0),), dtype=torch.float32)  # Dummy attribute
                parity = statistical_parity(logits.argmax(dim=1), protected_attribute)
                if parity > LargestPar:
                    LargestPar = parity
                    print(f"Statistical Parity: {parity}")

            if i == 0:
                print("Loss before:", loss.item())
                
            if i == 9:
                print("Loss after:", loss.item())
                
    except Exception as e:
        logging.error(f'Error during training: {e}')
    
    end_time = time.time()
    training_duration = end_time - start_time
    print(f"Training time: {training_duration:.2f} seconds")
    
    # Timing the testing process
    start_time = time.time()
    
    try:
        correct = 0
        total = 0
        with torch.no_grad():
            
            LargestPar = 0
            
            for batch, labels in test_loader:
                batch = batch.flatten(1)
                outputs = model(batch)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Statistical parity fairness check (dummy example)
                protected_attribute = torch.randint(0, 2, (batch.size(0),), dtype=torch.float32)  # Dummy attribute
                parity = statistical_parity(predicted, protected_attribute)
                if parity > LargestPar:
                    LargestPar = parity
                    print(f"Statistical Parity: {parity}")

        accuracy = 100 * correct / total
        print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

    except Exception as e:
        logging.error(f'Error during testing: {e}')
    
    end_time = time.time()
    testing_duration = end_time - start_time
    print(f"Testing time: {testing_duration:.2f} seconds")
