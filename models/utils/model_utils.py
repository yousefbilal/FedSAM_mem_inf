import json
import numpy as np
import os
from collections import defaultdict


def batch_data(data, batch_size, seed):
    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = data["x"]
    data_y = data["y"]
    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(np.asarray(data_x))
    np.random.set_state(rng_state)
    np.random.shuffle(np.asarray(data_y))

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        yield (batched_x, batched_y)


def read_dir(data_dir, alpha=None):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    if alpha is None:
        files = [f for f in files if f.endswith(".json")]
    else:
        alpha = "alpha_{:.2f}".format(alpha)
        files = [f for f in files if f.endswith(alpha + ".json")]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        data.update(cdata["user_data"])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir, alpha=None):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    train_data = os.listdir(train_data_dir)

    train_labels = list(
        map(lambda x: int(os.path.splitext(x)[0].split("_")[-1]), train_data)
    )
    train_data = {"x": np.array(train_data), "y": np.array(train_labels)}
    test_data_files = os.listdir(test_data_dir)
    test_labels = list(
        map(lambda x: int(os.path.splitext(x)[0].split("_")[-1]), test_data_files)
    )

    test_data = defaultdict(lambda: None)
    
    test_data.update(
        {"100": {"x": np.array(test_data_files), "y": np.array(test_labels)}}
    )

    test_clients = ["100"]

    return test_clients, train_data, test_data


def combine_client_data(data_dict):
    all_x = []
    all_y = []

    for client_id in data_dict:
        all_x.append(data_dict[client_id]["x"])
        all_y.append(data_dict[client_id]["y"])

    combined_x = np.concatenate(all_x, axis=0)
    combined_y = np.concatenate(all_y, axis=0)

    return combined_x, combined_y


def split_noniid_dirichlet(combined_x, combined_y, n_clients, alpha=0.5):
    """
    Split combined data into clients using Dirichlet distribution.

    Args:
        combined_x: Combined features array of all data
        combined_y: Combined labels array of all data
        n_clients: number of clients
        alpha: concentration parameter (smaller = more non-IID)
               - alpha -> 0: each client gets 1-2 classes (highly non-IID)
               - alpha = 0.5: moderate non-IID
               - alpha = 1.0: mild non-IID
               - alpha -> ∞: approaches IID

    Returns:
        defaultdict: Dictionary where each key is client_id and value is dict with 'x' and 'y' data
    """
    n_classes = len(np.unique(combined_y))
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    # Get indices for each class
    class_indices = [np.where(combined_y == i)[0] for i in range(n_classes)]
    client_indices = [[] for _ in range(n_clients)]

    # Distribute indices according to Dirichlet distribution
    for class_idx, indices in enumerate(class_indices):
        np.random.shuffle(indices)
        proportions = label_distribution[class_idx]
        proportions = proportions / proportions.sum()  # normalize
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

        for client_id, idx_range in enumerate(np.split(indices, split_points)):
            client_indices[client_id].extend(idx_range)

    # Shuffle each client's indices
    for client_id in range(n_clients):
        np.random.shuffle(client_indices[client_id])

    # Create client data defaultdict
    client_data = defaultdict(lambda: None)
    for client_id in range(n_clients):
        indices = client_indices[client_id]
        client_data[str(client_id)] = {
            "x": combined_x[indices],
            "y": combined_y[indices],
        }

    return client_data
