from FedAvgTest import fedavg_test
import numpy as np
import matplotlib.pyplot as plt

model_files = ['final_model_cfa1001.py', 'Cfar10Model.py']
learning_rates = [0.053081, 0.0001]
dir_name = 'best_model_dirichlet_01'
t = np.arange(0, 100)
# Initialize the plot
plt.figure()

# Loop over model_files and learning_rates to calculate and plot test_acc
for i in range(len(model_files)):
    print(str(i) + ' th model')
    test_loss, test_acc = fedavg_test(model_files[i], learning_rates[i])

    # Plot the test accuracy for this model
    plt.plot(t, test_acc, label=f'Model {str(model_files[i])}')  # Label each plot with the model number

    np.save(dir_name + '/test_accuracy_' + str(i), np.array(test_acc))
# Add titles and labels
plt.title('Test Accuracy over Time for Each Model')
plt.xlabel('Time')
plt.ylabel('Test Accuracy')

# Add a legend to distinguish the plots
plt.legend()

# Show the plot
plt.show()


