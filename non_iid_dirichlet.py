import numpy as np


def non_iid_dirichlet_split(dataset, num_clients, alpha, num_classes):
    # Initialize the data distribution for each client
    client_data_indices = {i: [] for i in range(num_clients)}
    client_class_counts = np.zeros((num_clients, num_classes), dtype=int)
    # Get the targets (labels) for the dataset
    targets = np.array(dataset.targets)

    # For each class, split the data across clients using a Dirichlet distribution
    for class_idx in range(num_classes):
        # Get indices of all samples belonging to this class
        class_indices = np.where(targets == class_idx)[0]

        # Shuffle the class indices
        np.random.shuffle(class_indices)

        # Sample from Dirichlet distribution for splitting among clients
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Proportionally distribute the data to each client
        proportions = np.cumsum(proportions) * len(class_indices)
        proportions = proportions.astype(int)
        print(np.sum(proportions))
        # Assign data to each client
        start_idx = 0
        for client_idx, end_idx in enumerate(proportions):
            assigned_indices = class_indices[start_idx:end_idx]
            print('assined  ' + str(assigned_indices))
            client_data_indices[client_idx] += class_indices[start_idx:end_idx].tolist()
            client_class_counts[client_idx, class_idx] = len(assigned_indices)
            start_idx = end_idx

    return client_data_indices, client_class_counts
