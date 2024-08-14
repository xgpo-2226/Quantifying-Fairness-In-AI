# Import necessary modules and libraries
import torch
import multiprocessing
from multiprocessing import set_start_method, freeze_support
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from ucimlrepo import fetch_ucirepo  # Module for fetching datasets from UCI ML repository
from typing import Callable
# import logging  # Standard Python logging module while program runs
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler  # Module for data normalization
from sklearn.metrics import f1_score  # to calculat f1 score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, hamming_loss, auc # in-built AI analysis metrics
from sklearn.preprocessing import MultiLabelBinarizer, label_binarize # to binarise the labels/output of the model
# from sklearn.metrics import plot_precision_recall_curve
from scipy.stats import chisquare, chi2_contingency
from sklearn.utils import resample # to resample protected attributes appropriately
import matplotlib.pyplot as plt  # for graph plotting
from scipy.stats import wasserstein_distance  # to calculate Wasserstein distance (measure difference between distributions)
import openpyxl  # writing to excel
from imblearn.over_sampling import SMOTE  # for batch normalisation
from imblearn.over_sampling import ADASYN  # Alternative to SMOTE for imbalanced datasets
import academic_success_equal_odds as EQO
import academic_success_intersx_parity as IXP


device = 'cpu'

# Function to initialize workers for data loading
def worker_init_fn(worker_id):
    logging.debug(f"Worker {worker_id} initializing")

# Model definition of a Multi-Layer Perceptron (MLP) model with multiple hidden layers and dropout
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layer_size: int, hidden_layers: int, output_size: int,
                 activation_fn: Callable[[], nn.Module] = nn.ReLU, dropout_rate: float = 0.3):
        super(MLP, self).__init__()
        layers = []  # List to hold the layers

        # Input layer
        layers.append(nn.Linear(input_size, hidden_layer_size))  # Input layer (determines weights and biases)
        layers.append(nn.BatchNorm1d(hidden_layer_size))  # Batch Normalization
        layers.append(activation_fn())  # Activation function for the input layer (using LeakyReLU)
        layers.append(nn.Dropout(dropout_rate))  # Dropout layer for regularization

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer_size, hidden_layer_size))  # Hidden layers
            layers.append(nn.BatchNorm1d(hidden_layer_size))  # Batch Normalization
            layers.append(activation_fn())  # Activation function for hidden layers
            layers.append(nn.Dropout(dropout_rate))  # Dropout layer for regularization

        # Outer layer
        layers.append(nn.Linear(hidden_layer_size, output_size))  # Output layer

        # Combining all layers to make the model
        self.model = nn.Sequential(*layers)  # Sequential model composed of the layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)  # Forward pass through the model

# Function to calculate statistical parity
def statistical_parity(predictions, protected_attribute, alpha=0.5):
    group_0 = predictions[protected_attribute == 0]  # Predictions for group 0
    group_1 = predictions[protected_attribute == 1]  # Predictions for group 1
    parity = torch.abs(group_0.mean() - group_1.mean())  # Absolute difference in means
    
    # Calculate the median for each group
    group_0_median = torch.median(group_0)
    group_1_median = torch.median(group_1)
    
    return alpha * parity.item() # Return the statistical parity value

# Main code block
if __name__ == "__main__":
    # Ensure safe multiprocessing start method
    set_start_method('spawn', force=True)
    freeze_support()

    # Hyperparameters
    batch_size = 128
    num_workers = 0 # 16 is the maximum no. CPUs that on current laptop
    learning_rate = 1e-3
    hidden_layer_size = 85
    hidden_layers = 5
    num_epochs = 11 # 30
    dropout_rate = 0.3
    
    # multiprocessing.cpu_count()

    # Fetch the new dataset from UCI ML repository
    dataset = fetch_ucirepo(id=697)  # ID for the new dataset

    # Debug: Print fetched dataset details
    print("Fetched dataset:", dataset)

    # Ensure dataset is valid
    if dataset is None or not hasattr(dataset, 'data'):
        raise ValueError("Dataset could not be fetched or is invalid.")

    # Combine features and targets into a single DataFrame
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # Debug: Print first few rows of the DataFrame (5 by default)
    print("DataFrame head:\n", df.head())

    # Encoding categorical features and target (into dummy/indicator variables)
    df = pd.get_dummies(df)

    # Assuming 'Target' is the target and 'Gender' is the protected attribute
    features = df.drop(columns=['Target_Dropout', 'Target_Enrolled', 'Target_Graduate', 'Gender']).values  # Exclude target and protected attribute
    labels = df[['Target_Dropout', 'Target_Enrolled', 'Target_Graduate']].values  # Example: using 'Dropout' as labels

    print(f"labels: {labels}")

    protected_attr = df['Gender'].values  # Gender column as protected attribute (0 for Female, 1 for Male)

    print("Label distribution in training set:")
    print(pd.DataFrame(labels).sum())

    # # Balance the dataset using ADASYN
    # smote = SMOTE()
    # features, labels = smote.fit_resample(features, labels)
    adasyn = ADASYN()
    features, labels = adasyn.fit_resample(features, labels)

    # Resample protected attributes accordingly
    protected_attr_resampled = resample(protected_attr, n_samples=features.shape[0], random_state=42)

    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Convert features, labels, and protected attribute to PyTorch tensors
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)  # Use float32 for multi-label
    protected_attr_tensor = torch.tensor(protected_attr_resampled, dtype=torch.float32).to(device)

    # Check sizes of tensors
    print(f"Features tensor size: {features_tensor.size()}")
    print(f"Labels tensor size: {labels_tensor.size()}")
    print(f"Protected attribute tensor size: {protected_attr_tensor.size()}")

    # Ensure that tensors have the same number of samples
    assert features_tensor.size(0) == labels_tensor.size(0) == protected_attr_tensor.size(0), \
        "Size mismatch between tensors"

    # Create TensorDataset from the above tensors
    dataset = TensorDataset(features_tensor, labels_tensor, protected_attr_tensor)


    # Split dataset into train and test sets (80-20 split)
    train_size = int(0.6 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Initialize a list to store the indices of test set entries in the original dataset
    test_indices = []
    
    # Iterate over the test dataset to find the corresponding indices in the original dataset
    for test_data in test_dataset:
        for idx, original_data in enumerate(dataset):
            if torch.equal(test_data[0], original_data[0]) and torch.equal(test_data[1], original_data[1]) and torch.equal(test_data[2], original_data[2]):
                test_indices.append(idx)
                break    

    # DataLoaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Instantiate the MLP model
    input_size = features.shape[1]  # Number of input features
    output_size = labels.shape[1]  # Number of output classes
    model = MLP(input_size, hidden_layer_size, hidden_layers, output_size, activation_fn=nn.LeakyReLU, dropout_rate=dropout_rate).to(device)

    # Define optimiser (AdamW) and loss function (CrossEntropyLoss)
    # optimiser = optim.SGD(model.parameters(), lr=learning_rate)
    optimiser = optim.AdamW(model.parameters(), lr=learning_rate)

    label_counts = pd.DataFrame(labels, columns=['Dropout', 'Enrolled', 'Graduate']).sum()
    class_counts = label_counts.values
    class_weights = 1.0 / class_counts
    # class_weights_i = 3 / sum(class_counts) 
    # class_weights = [class_weights_i,class_weights_i,class_weights_i]
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    loss_function = nn.CrossEntropyLoss(weight=weights_tensor)
    # loss_function = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification

    # Learning rate scheduler to adjust learning rate during training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.1, patience=3)

    # Training loop with timing and statistical parity
    try:
        training_times = []
        start_time = time.time()
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1} out of {num_epochs}")
            start_epoch_time = time.time()

            all_predictions = []
            all_labels = []
            all_protected_attrs = []

            model.train()  # Set the model to training mode

            for batch in train_loader:
                features_batch, labels_batch, protected_attr_batch = batch
                optimiser.zero_grad()  # zero the parameter gradients of the optimiser
                logits = model(features_batch)  # forward pass to get raw output values

                predictions = F.softmax(logits, dim=1)
                _, predicted_indices = torch.max(logits, 1)
                predictions = torch.zeros_like(logits)
                predictions.scatter_(1, predicted_indices.unsqueeze(1), 1)

                classification_loss = loss_function(logits, labels_batch)
                parity_loss = statistical_parity(logits, protected_attr_batch)  # loss from
                loss = classification_loss + parity_loss + np.random.randint(100)
                
                # loss = loss_function(logits, labels_batch) # compare raw output with expected classes to find difference
                loss.backward() # backward pass
                optimiser.step() # update model parameters
    
                # Collect predictions for statistical parity
                predictions = torch.sigmoid(logits) > 0.5  # Sigmoid and threshold for binary predictions
                all_predictions.extend(predictions.detach().cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())
                all_protected_attrs.extend(protected_attr_batch.cpu().numpy())
    
            # Timing the epoch
            elapsed_time = time.time() - start_epoch_time
            training_times.append(elapsed_time)
            print(f"\nEpoch time: {elapsed_time:.2f} seconds")
            
            # Update the learning rate
            scheduler.step(loss)
    
        # Evaluate accuracy after training
        end_time = time.time()
        training_duration = end_time - start_time
        print(f"\nTraining time: {training_duration:.2f} seconds")
    
        ## TESTING
    
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        fairness_correct = 0
        protected_attribute_positive = 0
        protected_attribute_negative = 0
        
        testing_times = []
        all_predictions = []
        all_features = []
        all_labels = []
        all_protected_attrs = []
        all_pred_probs = []
        # parity = 'Invalid'
        
        fpr = [[],[],[]]
        tpr = [[],[],[]]
        
        # Timing the testing process
        start_time = time.time()
        
        with torch.no_grad(): # Disable gradient calculations for inference
            
            for features_batch, labels_batch, protected_attr_batch in test_loader:
                
                start_train_batch_time = time.time()
                
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device)
                protected_attr_batch = protected_attr_batch.to(device)
                
                
                logits = model(features_batch)
                predictions = torch.sigmoid(logits) > 0.5  # Sigmoid and threshold for binary predictions
                pred_probs = F.softmax(logits, dim=1)  # Store softmax probabilities
                
                # Ensure single '1' per prediction
                _, predicted_indices = torch.max(logits, 1)
                predictions = torch.zeros_like(logits)
                predictions.scatter_(1, predicted_indices.unsqueeze(1), 1)
         
                
                # parity = statistical_parity(predictions, protected_attr_batch)
                parity = statistical_parity(logits, protected_attr_batch) ##!
                
                # Convert tensors to numpy arrays for comparison
                predicted_np = predictions.cpu().numpy()
                labels_np = labels_batch.cpu().numpy()
                features_np = features_batch.cpu().numpy()
                pred_probs_np = pred_probs.cpu().numpy()  # Convert probabilities to numpy array
        
                # Compare each row (list of predictions) as a whole
                for pred, label in zip(predicted_np, labels_np):
                    total += 1
                    if np.array_equal(pred, label):
                        correct += 1
                
                all_predictions.append(predicted_np)
                all_labels.append(labels_np)
                all_protected_attrs.append(protected_attr_batch.cpu().numpy())
                all_features.append(features_np)
                all_pred_probs.append(pred_probs_np)  # Store probabilities 
                    
                # Timing the test batch
                elapsed_time = time.time() - start_train_batch_time
                testing_times.append(elapsed_time)
                
                # Convert one-hot encoded labels to class labels
                all_labels_class = np.argmax(labels_np, axis=1)
                all_predictions_class = np.argmax(predicted_np, axis=1)
                
                cm = confusion_matrix(all_labels_class, all_predictions_class)
                
                # Calculate FPR for each class
                num_classes = cm.shape[0]
                
                for i in range(num_classes):
                    # True positives, false positives, false negatives, true negatives
                    tp = cm[i, i]
                    fp = cm[i, :].sum() - tp # type 1 error (reject null hypo)
                    fn = cm[:, i].sum() - tp # type 2 error
                    tn = cm.sum() - (tp + fp + fn)
                    
                    # False positive rate: FP / (FP + TN)
                    fpr_i = fp / (fp + tn) # 1 - specificity
                    fpr[i].append(fpr_i)
                    
                    # True positive rate: TP / (FN + TP)
                    tpr_i = tp / (fn + tp) # sensitivity/recall | rate of actual positives that are predicted as positive
                    tpr[i].append(tpr_i)
                    
                    # Precision: TP / (FP + TP) | how many predicted postives are actually positive
                    
                    # True negative rate/specificity or fallout: TN / (FP + TN)
                    
    
    
        end_time = time.time()
        testing_duration = end_time - start_time
        print(f"\nTesting time: {testing_duration:.2f} seconds")
        
        accuracy = 100 * correct / total # (TP+TN)/(TP+TN+FP+FN)
        print(f'\n\nAccuracy of the network on the test set: {accuracy:.3f}%')
        
        # Convert lists to numpy arrays
        all_predictions_np = np.vstack(all_predictions)
        all_labels_np = np.vstack(all_labels)
        all_protected_attrs_np = np.hstack(all_protected_attrs)
        all_pred_probs_np = np.vstack(all_pred_probs)
        all_features_np = np.vstack(all_features)
        
        print(f'\nAccuracy Score is: {accuracy_score(all_labels_np,all_predictions_np):.4f}')
        
        # parity = statistical_parity(torch.from_numpy(all_predictions_np), torch.from_numpy(all_protected_attrs_np))
        parity = statistical_parity(model(torch.from_numpy(all_features_np)), torch.from_numpy(all_protected_attrs_np))
        print(f"\nStatistical Parity: {parity:.5f}")
        
        # equalised_odds_value = EQO.equalised_odds(torch.from_numpy(all_predictions_np), torch.from_numpy(all_labels_np), torch.from_numpy(all_protected_attrs_np))
        equalised_odds_value = EQO.equalised_odds(model(torch.from_numpy(all_features_np)), torch.from_numpy(all_labels_np), torch.from_numpy(all_protected_attrs_np))
        print(f"\nEqualised Odds Rate: {equalised_odds_value:.5f}") 
        
        dataset = fetch_ucirepo(id=697)  # ID for the new dataset
        df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        df = pd.get_dummies(df)
        sens_attr = df[['Gender', 'Nacionality', 'Educational special needs']].values # instantiate a new list of sensitive attributes
        sens_attr_resampled = resample(sens_attr, n_samples=labels.shape[0], random_state=42)
        sens_attr_tensor = torch.tensor(sens_attr_resampled, dtype=torch.float32).to(device)
        parity_intx = IXP.intersectional_statistical_parity(model(torch.from_numpy(all_features_np)), sens_attr_tensor[test_indices])
        print(f"\nIntersection Statistical Parity: {parity_intx:.5f}") 
        
        f1 = f1_score(all_labels_np, all_predictions_np, average='macro')
        print(f'\nF1-Score: {f1:.3f}')
        
        # # Scatter plot of actual vs predicted results for sensitive groups
        # actual_graduate_male = all_labels_np[all_protected_attrs == 1][:, 2]
        # predicted_graduate_male = all_predictions_np[all_protected_attrs == 1][:, 2]
        # actual_graduate_female = all_labels_np[all_protected_attrs == 0][:, 2]
        # predicted_graduate_female = all_predictions_np[all_protected_attrs == 0][:, 2]
        
        # # Get indices where the elements in all_protected_attrs are equal to 1
        # indices_male = np.where(all_protected_attrs_np[0] == 1)[0]
        # # Get indices where the elements in all_protected_attrs are equal to 0
        # indices_female = np.where(all_protected_attrs_np[0] == 0)[0]
        
        # Get indices where the elements in all_protected_attrs are equal to 1
        indices_male = np.where(all_protected_attrs_np == 1)[0]
        # Get indices where the elements in all_protected_attrs are equal to 0
        indices_female = np.where(all_protected_attrs_np == 0)[0]
        
        # Index into all_labels_np using the indices to get the corresponding rows
        actual_graduate_male = all_labels_np[indices_male]
        predicted_graduate_male = all_predictions_np[indices_male]
        actual_graduate_female = all_labels_np[indices_female]
        predicted_graduate_female = all_predictions_np[indices_female]
        
        # finds the index of the column in each row of the '1' value
        actual_graduate_male_bin = np.argmax(actual_graduate_male, axis=1)
        predicted_graduate_male_bin = np.argmax(predicted_graduate_male, axis=1)
        actual_graduate_female_bin = np.argmax(actual_graduate_female, axis=1)
        predicted_graduate_female_bin = np.argmax(predicted_graduate_female, axis=1) 
        
        
        # Epoch Runtime/Performance Time Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs + 1), training_times, label='Training Time')
        # plt.plot(range(1, num_epochs + 1), testing_times, label='Testing Time')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Epoch Runtime/Performance Time:\n STATISTICAL PARITY FAIRNESS')
        plt.legend()
        plt.show()
    
        
        all_labels_np = all_labels_np.astype(int)
        all_predictions_np = all_predictions_np.astype(int) #!
        
        
        # Convert numpy arrays to lists of lists
        all_labels_list = [list(label) for label in all_labels_np]
        all_predictions_list = [list(prediction) for prediction in all_predictions_np]
        
        mlb = MultiLabelBinarizer() #!
        
        all_labels_bin = np.argmax(all_labels_np, axis=1)
        all_predictions_bin = np.argmax(all_predictions_np, axis=1)
        
        # Compute the classification report
        print("\n\nClassification Report:")
        # print(classification_report(all_labels_bin, all_predictions_bin, target_names=mlb.classes_.astype(str)))
        print(classification_report(all_labels_bin, all_predictions_bin))
        
        # Compute Hamming loss (proportion of incorrectly classified labels, 0 - 1 possible value range)
        print("\n\nHamming Loss:", hamming_loss(all_labels_bin, all_predictions_bin))
        
    
        # Convert one-hot encoded labels to class labels
        all_labels_class = np.argmax(all_labels_np, axis=1)
        all_predictions_class = np.argmax(all_predictions_np, axis=1)
        
        # Compute the confusion matrix
        cm = confusion_matrix(all_labels_class, all_predictions_class)
        print("\nConfusion Matrix:")
        print(cm)
        
        # # Calculate FPR for each class
        num_classes = cm.shape[0]
        fpr = []
        tpr = []
        labels_str = ['Dropout', 'Enrolled', 'Graduate']
        
        for i in range(num_classes):
            # True positives, false positives, false negatives, true negatives
            tp = cm[i, i]
            fp = cm[i, :].sum() - tp # type 1 error (reject null hypo)
            fn = cm[:, i].sum() - tp # type 2 error
            tn = cm.sum() - (tp + fp + fn)
            
            # 544 / (544)
            
            # False positive rate: FP / (FP + TN)
            fpr_i = fp / (fp + tn) # 1 - specificity
            fpr.append(fpr_i)
            
            # True positive rate: TP / (FN + TP)
            tpr_i = tp / (fn + tp) # sensitivity/recall | rate of actual positives that are predicted as positive
            tpr.append(tpr_i)
            
            # Precision: TP / (FP + TP) | how many predicted postives are actually positive
            
            # True negative rate/specificity or fallout: TN / (FP + TN)
            
        
        for i in range(len(fpr)):
            print(f"\nThe False Positive Rate for the label '{labels_str[i]}' is: {fpr[i]}")
            print(f"The True Positive Rate for the label '{labels_str[i]}' is: {tpr[i]}")
            
        
        # Binarize the labels for one-vs-rest classification
        y_test_binarized = label_binarize(all_labels_np, classes=[0, 1, 2])
        
        
        n_classes = all_labels_np.shape[1]

        # Initialize TPR and FPR arrays
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(all_labels_np[:,i], all_predictions_np[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        
        from itertools import cycle
        # Plotting
        plt.figure()
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC):\n STATISTICAL PARITY FAIRNESS')
        plt.legend(loc="lower right")
        plt.show()

        
        # Compute the ROC AUC score using "macro" or "micro" averaging
        roc_auc_macro = roc_auc_score(y_test_binarized, all_predictions_np, average='macro')
        roc_auc_micro = roc_auc_score(y_test_binarized, all_predictions_np, average='micro')
        
        print(f"\nROC AUC Score (Macro-Averaged): {roc_auc_macro:.5f}")
        print(f"ROC AUC Score (Micro-Averaged): {roc_auc_micro:.5f}")
        
        # Precision-Recall Curve
        # disp = plot_precision_recall_curve(model, X_test, all_labels)
        # disp = precision_recall_curve(model, all_features_np, all_labels_np)
        # Loop through each class to compute precision-recall curve
        for i in range(all_labels_np.shape[1]):
            precision, recall, _ = precision_recall_curve(all_labels_np[:, i], all_predictions_np[:, i])
            
            # Plot the precision-recall curve for each class
            plt.plot(recall, precision, lw=2, label=f'{labels_str[i]}')
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend(loc='best') # automatically place wherever legend interferes with graph the least
        plt.title('Precision-Recall Curve | STATISTICAL PARITY FAIRNESS')
        plt.show()
        
        actual_male_counts = np.bincount(actual_graduate_male_bin)
        predicted_male_counts = np.bincount(predicted_graduate_male_bin)
        
        actual_female_counts = np.bincount(actual_graduate_female_bin)
        predicted_female_counts = np.bincount(predicted_graduate_female_bin)
        
        chi_square_male = chisquare(f_obs=actual_male_counts, f_exp=predicted_male_counts)
        chi_square_female = chisquare(f_obs=actual_female_counts, f_exp=predicted_female_counts)
        
        print("\nMeasuring ACCURACY with CHI-Square")
        print(f"Chi-Square Test for Male Group: {chi_square_male}") # statistic = the chi squared distance
        print(f"Chi-Square Test for Female Group: {chi_square_female}") # pvalue < 0.05 = statistically sig difference
        
        print("\nMeasuring FAIRNESS with CHI-Square")
        # print(f"Chi-Square Test Across All Groups: {chi_square_all}")
        
        # Create a contingency table
        contingency_table = np.array([predicted_male_counts, predicted_female_counts])
        
        # Perform the Chi-Square test for homogeneity
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Output the results
        print(f"Chi-Sqaure Statistic Across All Groups: {chi2_stat:.5f}")
        print(f"P-Value Across All Groups: {p_value:.5f}")
    
    except Exception as e:
        logging.error(f'Error during testing: {e}')
