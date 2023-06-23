import matplotlib.pyplot as plt
import math
from model import train_losses, train_acc, test_losses, test_acc

#########################################################################################################################################################
def return_dataset_images(train_loader, total_images, labels_mapping):
    """
        This function prints images from train loader.
        Params : 
        train_loader : training data loader
        total_images : no. of images to be printed
        labels_mapping : Mapping of labels
    """
    batch_data, batch_label = next(iter(train_loader)) 

    fig = plt.figure()

    for i in range(total_images):
        plt.subplot(math.ceil(total_images/5), 5,i+1)
        #plt.tight_layout()
        img = batch_data[i].numpy()     # Shape 3x32x32
        img = img.transpose(1,2,0)      # Shape 32x32x3
        label = labels_mapping[batch_label[i].item()]
        plt.imshow(img)
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
#########################################################################################################################################################

#########################################################################################################################################################
def draw_graphs():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
#########################################################################################################################################################

def plot_incorrect_predictions(predictions, class_map, count=10):
    """Plot Incorrect predictions

    Args:
        predictions (list): List of all incorrect predictions
        class_map (dict): Lable mapping
        count (int, optional): Number of samples to print, multiple of 5. Defaults to 10.
    """
    print(f'Total Incorrect Predictions {len(predictions)}')

    if not count % 5 == 0:
        print("Count should be multiple of 10")
        return

    classes = list(class_map.values())

    fig = plt.figure(figsize=(10, 5))
    for i, (d, t, p, o) in enumerate(predictions):
        ax = fig.add_subplot(int(count/5), 5, i + 1, xticks=[], yticks=[])
        ax.set_title(f'{classes[t.item()]}/{classes[p.item()]}')
        plt.imshow(d.cpu().numpy().transpose(1, 2, 0))
        if i+1 == 5*(count/5):
            break