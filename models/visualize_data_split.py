import numpy as np
import matplotlib.pyplot as plt
from utils.model_utils import read_data, split_noniid_dirichlet
import argparse

def visualize_client_split(client_data, n_classes=10, output_dir='plots'):
    """
    Visualizes the data distribution across clients.
    
    Args:
        client_data: Dictionary of client data from split_noniid_dirichlet
        n_classes: Number of classes in the dataset
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    n_clients = len(client_data)
    
    # Collect statistics
    client_ids = sorted([int(k) for k in client_data.keys()])
    client_sizes = []
    client_class_distributions = []
    
    for client_id in client_ids:
        client_str = str(client_id)
        labels = client_data[client_str]['y']
        client_sizes.append(len(labels))
        
        # Count samples per class
        class_counts = np.bincount(labels, minlength=n_classes)
        client_class_distributions.append(class_counts)
    
    client_class_distributions = np.array(client_class_distributions)
    
    # ===== PLOT 1: Heatmap of Client-Class Distribution =====
    fig, ax = plt.subplots(figsize=(14, max(8, n_clients * 0.15)))
    
    im = ax.imshow(client_class_distributions, aspect='auto', cmap='YlOrRd')
    
    # Set ticks
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_clients))
    ax.set_xticklabels([f'Class {i}' for i in range(n_classes)])
    ax.set_yticklabels([f'Client {i}' for i in client_ids])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Number of Samples', rotation=270, labelpad=20)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Client ID', fontsize=12)
    ax.set_title('Client-Class Data Distribution Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/client_class_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/client_class_heatmap.pdf', bbox_inches='tight')
    plt.show()
    
    # ===== PLOT 2: Stacked Bar Chart =====
    fig, ax = plt.subplots(figsize=(max(12, n_clients * 0.3), 6))
    
    # Prepare data for stacked bar
    bottom = np.zeros(n_clients)
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for class_id in range(n_classes):
        class_counts = client_class_distributions[:, class_id]
        ax.bar(client_ids, class_counts, bottom=bottom, 
               label=f'Class {class_id}', color=colors[class_id])
        bottom += class_counts
    
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Class Distribution per Client (Stacked)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/client_class_stacked.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/client_class_stacked.pdf', bbox_inches='tight')
    plt.show()
    
    # ===== PLOT 3: Client Size Distribution =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(client_ids, client_sizes, color='steelblue', alpha=0.7)
    ax.axhline(np.mean(client_sizes), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(client_sizes):.1f}')
    
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Total Samples per Client', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/client_sizes.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/client_sizes.pdf', bbox_inches='tight')
    plt.show()
    
    # ===== PLOT 4: Class Diversity per Client =====
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count how many classes each client has (non-zero counts)
    n_classes_per_client = np.sum(client_class_distributions > 0, axis=1)
    
    ax.bar(client_ids, n_classes_per_client, color='darkorange', alpha=0.7)
    ax.axhline(np.mean(n_classes_per_client), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(n_classes_per_client):.2f}')
    
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_ylabel('Number of Unique Classes', fontsize=12)
    ax.set_title('Class Diversity per Client', fontsize=14, fontweight='bold')
    ax.set_yticks(range(0, n_classes + 1))
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/client_class_diversity.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/client_class_diversity.pdf', bbox_inches='tight')
    plt.show()
    
    # ===== Print Statistics =====
    print("\n" + "="*60)
    print("CLIENT DATA SPLIT STATISTICS")
    print("="*60)
    print(f"Total Clients: {n_clients}")
    print(f"Total Classes: {n_classes}")
    print(f"Total Samples: {sum(client_sizes)}")
    print(f"\nSamples per Client:")
    print(f"  Mean: {np.mean(client_sizes):.2f}")
    print(f"  Std:  {np.std(client_sizes):.2f}")
    print(f"  Min:  {np.min(client_sizes)}")
    print(f"  Max:  {np.max(client_sizes)}")
    print(f"\nClasses per Client:")
    print(f"  Mean: {np.mean(n_classes_per_client):.2f}")
    print(f"  Std:  {np.std(n_classes_per_client):.2f}")
    print(f"  Min:  {np.min(n_classes_per_client)}")
    print(f"  Max:  {np.max(n_classes_per_client)}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Visualize federated data split')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset name')
    parser.add_argument('--n_clients', type=int, default=100,
                        help='Number of clients')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha parameter (0 for one-class-per-client)')
    parser.add_argument('--n_classes', type=int, default=10,
                        help='Number of classes in dataset')
    parser.add_argument('--output_dir', type=str, default='plots/data_split',
                        help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    train_data_dir = f'../data/{args.dataset}/data/img/train'
    test_data_dir = f'../data/{args.dataset}/data/img/test'
    
    print(f"Loading data from {train_data_dir}...")
    _, train_data, _ = read_data(train_data_dir, test_data_dir)
    
    combined_x = train_data['x']
    combined_y = train_data['y']
    
    print(f"Total training samples: {len(combined_x)}")
    print(f"Splitting into {args.n_clients} clients with alpha={args.alpha}...")
    
    # Split data
    client_data = split_noniid_dirichlet(
        combined_x, combined_y, 
        n_clients=args.n_clients, 
        alpha=args.alpha
    )
    
    # Visualize
    visualize_client_split(client_data, n_classes=args.n_classes, 
                          output_dir=args.output_dir)
    
    print(f"\nPlots saved to {args.output_dir}/")


if __name__ == '__main__':
    main()