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
    train_data = sorted(os.listdir(train_data_dir))

    train_labels = list(
        map(lambda x: int(os.path.splitext(x)[0].split("_")[-1]), train_data)
    )
    train_data = {"x": np.array(train_data), "y": np.array(train_labels)}
    test_data_files = sorted(os.listdir(test_data_dir))
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


# def split_noniid_dirichlet(combined_x, combined_y, n_clients, alpha=0.5):
#     """
#     Split combined data into clients using Dirichlet distribution.

#     Args:
#         combined_x: Combined features array of all data
#         combined_y: Combined labels array of all data
#         n_clients: number of clients
#         alpha: concentration parameter (smaller = more non-IID)
#                - alpha -> 0: each client gets 1-2 classes (highly non-IID)
#                - alpha = 0.5: moderate non-IID
#                - alpha = 1.0: mild non-IID
#                - alpha -> ∞: approaches IID

#     Returns:
#         defaultdict: Dictionary where each key is client_id and value is dict with 'x' and 'y' data
#     """
#     n_classes = len(np.unique(combined_y))
#     label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

#     # Get indices for each class
#     class_indices = [np.where(combined_y == i)[0] for i in range(n_classes)]
#     client_indices = [[] for _ in range(n_clients)]

#     # Distribute indices according to Dirichlet distribution
#     for class_idx, indices in enumerate(class_indices):
#         np.random.shuffle(indices)
#         proportions = label_distribution[class_idx]
#         proportions = proportions / proportions.sum()  # normalize
#         split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]

#         for client_id, idx_range in enumerate(np.split(indices, split_points)):
#             client_indices[client_id].extend(idx_range)

#     # Shuffle each client's indices
#     for client_id in range(n_clients):
#         np.random.shuffle(client_indices[client_id])

#     # Create client data defaultdict
#     client_data = defaultdict(lambda: None)
#     for client_id in range(n_clients):
#         indices = client_indices[client_id]
#         print(f"client{client_id}: {len(indices)}", end= "  ")
#         client_data[str(client_id)] = {
#             "x": combined_x[indices],
#             "y": combined_y[indices],
#         }

#     return client_data


def split_noniid_dirichlet(combined_x, combined_y, n_clients, alpha=0.5):
    """
    Splits data with two distinct behaviors:
    1. alpha = 0: STRICT "One Class Per Client" (Shared).
       - Deterministic.
       - Example: 100 clients, 10 classes -> Clients 0-9 share Class 0.
    2. alpha > 0: DIRICHLET distribution.
       - Guarantees all clients get equal amount of data.
       - Uses greedy backfilling to ensure 100% data usage.
    """
    n_classes = len(np.unique(combined_y))
    n_data = len(combined_y)

    # Organize indices by class
    class_indices = [np.where(combined_y == i)[0] for i in range(n_classes)]
    for indices in class_indices:
        np.random.shuffle(indices)

    client_data_indices = defaultdict(list)

    # ==========================================
    # CASE 1: Alpha = 0 (Deterministic Shards)
    # ==========================================
    if alpha == 0:
        # Determine which clients get which class
        # We split the client IDs into 'n_classes' groups
        client_groups = np.array_split(np.arange(n_clients), n_classes)

        for class_id, client_group in enumerate(client_groups):
            # The indices available for this class
            available_indices = class_indices[class_id]

            # Split this class's data among the clients in the group
            if len(client_group) > 0:
                shards = np.array_split(available_indices, len(client_group))

                for i, client_id in enumerate(client_group):
                    client_data_indices[client_id] = shards[i]

    # ==========================================
    # CASE 2: Alpha > 0 (Dirichlet Sampling)
    # ==========================================
    else:
        # 1. Calculate how many samples each client needs
        base_size = n_data // n_clients
        remainder = n_data % n_clients

        # 2. Generate preferences (Client-based logic)
        # Shape: (n_clients, n_classes)
        client_class_preferences = np.random.dirichlet([alpha] * n_classes, n_clients)

        # Track usage of class indices
        class_idx_pointers = np.zeros(n_classes, dtype=int)

        for client_id in range(n_clients):
            # Determine target size for this client
            target_size = base_size + (1 if client_id < remainder else 0)

            # Calculate demand per class based on preferences
            samples_per_class = (
                client_class_preferences[client_id] * target_size
            ).astype(int)

            # --- PHASE 1: Satisfy Preferences ---
            for class_id, count in enumerate(samples_per_class):
                if count == 0:
                    continue

                start = class_idx_pointers[class_id]
                available = len(class_indices[class_id]) - start
                take = min(count, available)

                if take > 0:
                    client_data_indices[client_id].extend(
                        class_indices[class_id][start : start + take]
                    )
                    class_idx_pointers[class_id] += take

            # --- PHASE 2: Backfill (Ensure Client is Full) ---
            # If preferences ran out of data, fill with whatever is left globally
            current_len = len(client_data_indices[client_id])
            needed = target_size - current_len

            if needed > 0:
                for class_id in range(n_classes):
                    if needed == 0:
                        break

                    start = class_idx_pointers[class_id]
                    available = len(class_indices[class_id]) - start

                    if available > 0:
                        take = min(needed, available)
                        client_data_indices[client_id].extend(
                            class_indices[class_id][start : start + take]
                        )
                        class_idx_pointers[class_id] += take
                        needed -= take

    # ==========================================
    # Finalize: Construct Dictionary
    # ==========================================

    client_data = defaultdict(lambda: None)
    for client_id, indices in client_data_indices.items():
        if len(indices) == 0:
            print(f"Warning: Client {client_id} is empty! (Check data availability)")
            continue

        indices = np.array(indices)
        np.random.shuffle(indices)  # Shuffle local data
        client_data[str(client_id)] = {
            "x": combined_x[indices],
            "y": combined_y[indices],
        }

    return client_data
