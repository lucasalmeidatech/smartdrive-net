import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from models import CustomDataset, CustomDatasetTest, CustomNet


if __name__ == '__main__':

   
    # Instantiate the neural network
    model = CustomNet().cuda()



    # Create custom dataset and dataloader
    data_dir = 'clipData/dataset/'  # Directory containing the combined .npz files for each view
    test_dir = 'clipData/trainTest/'  # Directory containing the combined .npz files for each view
    #labels_file = 'path_to_your_labels_file.npz'  # File containing the labels
    dataset = CustomDataset(data_dir)
    testDataset = CustomDatasetTest(test_dir)
    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders for train and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(testDataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)



    unique_labels, class_counts = dataset.get_class_counts()
    print("Unique labels:", unique_labels)
    print("Class counts:", class_counts)

    total = sum(class_counts)
    class_weights = [total / class_counts[i] for i in range(len(class_counts))]
    class_weights_tensor = torch.FloatTensor(class_weights).cuda()
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()#weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=1e-4) #1e-5 is a good starting point



    num_epochs = 6

    # Learning rate scheduler
    from torch.optim.lr_scheduler import OneCycleLR
    scheduler = OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=num_epochs)

    # Add early stopping parameters
    patience = 3  # Number of epochs to wait for improvement before stopping
    epochs_no_improve = 0  # Counter for epochs without improvement

    # Training loop

    best_val_loss = float('inf')
    best_val_acc = 0
    best_test_acc = 0
    best_test_epoch = 0
    best_val_epoch = 0

    from tqdm import tqdm
    model_path = 'clipData/bestModel.pth'  # Replace with the path where you want to save the model

    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for i, (inputs1, inputs2, inputs3, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs1 = inputs1.cuda()  # Move inputs to GPU
            inputs2 = inputs2.cuda()
            inputs3 = inputs3.cuda()
            labels = labels.cuda()  # Move labels to GPU
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs1, inputs2, inputs3)

            # Compute loss
            loss = criterion(outputs, labels.squeeze())  # Squeeze labels to remove extra dimension

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if i % 100 == 99:
                #print('[Train] [%d, %5d] loss: %.3f' %
                #      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        test_acc = 0
        with torch.no_grad():
            for inputs1, inputs2, inputs3, labels in val_dataloader:
                inputs1 = inputs1.cuda()  # Move inputs to GPU
                inputs2 = inputs2.cuda()
                inputs3 = inputs3.cuda()
                labels = labels.cuda()  # Move labels to GPU
                outputs = model(inputs1, inputs2, inputs3)
                _, predicted = torch.max(outputs, 1)
                #print('predictions',predicted)
                #print('labels',labels.squeeze())
                loss = criterion(outputs, labels.squeeze())
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()
            #input("Press Enter to continue...")

        # Adjust learning rate based on validation loss
        val_loss /= len(val_dataloader)
        scheduler.step(val_loss)

        val_losses.append(val_loss)
        val_accuracies.append(100 * correct / total)

        # If this epoch's validation loss is the best so far, save the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_epoch = epoch
            #torch.save(model, model_path)  # Replace with the path where you want to save the model
            #torch.save(model.state_dict(), '/home/cvrr/Desktop/CVPRchallenge/CVPRchallenge/clipData/bestStateDict2.pth')
            #print('Model saved')
            
        if best_val_acc < 100 * correct / total:
            best_val_acc = 100 * correct / total    
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1  # Increment the counter

        # If we've waited for patience epochs without improvement, stop training
        if epochs_no_improve == patience:
            print('Early stopping after %d epochs without improvement' % patience)
            break


        #print('[Epoch %d] Loss: %.4f | Validation loss: %.3f | Accuracy: %.2f %%' %
        #      (epoch + 1, loss, val_loss, 100 * correct / total))
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {running_loss/len(train_dataloader):.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {100 * correct / total:.2f}% | Best Val Acc: {best_val_acc:.2f}% at epoch {best_val_epoch}")
        print('-'*80)
        print('Current learning rate:', optimizer.param_groups[0]['lr'])
    #print('Finished Training')
        #print('Testing')
        # Test loop
        #model = torch.load(model_path)
        #model = model.cuda()  # Move model to GPU
        #model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        # Initialize lists to store the true and predicted labels
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for input1, input2, input3, labels in test_dataloader:
                input1 = input1.cuda()  # Move inputs to GPU
                input2 = input2.cuda()
                input3 = input3.cuda()
                labels = labels.cuda()  # Move labels to GPU
                outputs = model(input1, input2, input3)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()

                # Append the true and predicted labels to the lists
                true_labels.append(labels.squeeze().cpu().numpy())
                pred_labels.append(predicted.cpu().numpy())
        test_acc = 100 * correct / total
        if test_acc > best_test_acc:
            best_test_acc = 100 * correct / total
            best_test_epoch = epoch
            torch.save(model, model_path)  # Replace with the path where you want to save the model
            torch.save(model.state_dict(), 'clipData/bestStateDict2.pth')
            print('Model saved')
            import pandas as pd
            # Create a DataFrame from the true and predicted labels
            df = pd.DataFrame({
                'True Labels': np.concatenate(true_labels),
                'Predicted Labels': np.concatenate(pred_labels)
            })
            # Save the DataFrame as a CSV file, overwriting any existing file
            df.to_csv('data/examples/labels_and_predictions_2.csv', index=False)

            # Compute the confusion matrix
            cm = confusion_matrix(np.concatenate(true_labels), np.concatenate(pred_labels))

            # Normalize the confusion matrix
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Round the confusion matrix values to two decimal places
            cm = np.around(cm, decimals=2)

            # Increase figure size for better visibility
            fig, ax = plt.subplots(figsize=(20, 20))  # You can adjust the size as needed

            # Create the confusion matrix display object
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(16))

            # Use larger font for the numbers inside the boxes
            plt.rcParams.update({'font.size': 16})  # Adjust font size as needed

            # Plot the confusion matrix with color map
            disp.plot(cmap=plt.cm.Blues, values_format='.2f', ax=ax)

            # Save the confusion matrix as an image
            plt.savefig('data/examples/confusion_matrix_2.png', bbox_inches='tight')
            plt.close()
        #print('Test Accuracy: %.2f %%' % (100 * correct / total))
        print(f'Test Accuracy {test_acc}, current best: {best_test_acc} at epoch {best_test_epoch}')
        scheduler.step(test_acc)

    epochs_range = range(1, len(val_losses) + 1)

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('validation_loss_curve.png')
    plt.close()

    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, [acc for acc in val_accuracies], label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Over Time')
    plt.legend()
    plt.grid()
    plt.savefig('validation_accuracy_curve.png')
    plt.close()

    print("Gráficos salvos: 'validation_loss_curve.png' e 'validation_accuracy_curve.png'!")