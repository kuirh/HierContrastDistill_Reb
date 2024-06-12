import numpy as np
from matplotlib import pyplot as plt


def plot_output_and_label(label, output, filename):

    # Assuming 'output' and 'label' are PyTorch tensors, convert them to CPU and then to NumPy arrays for plotting
    output = output.cpu().detach().numpy()
    label = label.cpu().detach().numpy()

    ##
    with open(f'{filename}_data.csv', 'w') as f:
        f.write('Index,Label,Output\n')
        for i, (lbl, out) in enumerate(zip(label, output)):
            f.write(f'{i},{lbl},{out}\n')

    # Sort 'label' and use the indices to rearrange 'output' to match the sorted labels
    sorted_indices = np.argsort(label, axis=0).squeeze()
    sorted_label = label[sorted_indices]
    sorted_output = output[sorted_indices]

    # Generating a sequence of indices based on the length of the sorted label array
    indices = range(len(sorted_label))

    # Setting up the figure for plotting
    plt.figure(figsize=(8, 6))

    # Creating scatter plots for sorted output and label
    plt.scatter(indices, sorted_output, label='Output', alpha=0.6)
    plt.scatter(indices, sorted_label, label='Label', alpha=0.6)

    # Adding labels, title, and legend
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Output vs Label')
    plt.legend()

    # Saving the figure
    plt.savefig(f'{filename}.png')
    plt.close('all')
