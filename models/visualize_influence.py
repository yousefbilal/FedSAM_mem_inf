import numpy as np
import os
import argparse
import webbrowser
import sys
from utils.model_utils import read_data
import base64
import mimetypes

def embed_image_as_base64(image_path):
    """
    Reads an image file and returns it as a Base64-encoded
    data URI (a string that can be put in an <img> src tag).
    """
    try:
        # 1. Guess the MIME type (e.g., 'image/png')
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = "application/octet-stream" # Fallback

        # 2. Read the image file in binary mode
        with open(image_path, "rb") as f:
            image_binary_data = f.read()

        # 3. Encode it in Base64
        base64_data = base64.b64encode(image_binary_data)

        # 4. Decode the Base64 bytes into a UTF-8 string
        base64_string = base64_data.decode('utf-8')

        # 5. Format it as a data URI
        return f"data:{mime_type};base64,{base64_string}"

    except Exception as e:
        print(f"Error embedding image {image_path}: {e}")
        return "" # Return empty string on failure


def calculate_entropy(influence_vector, test_label, all_train_labels):
    """
    Calculates the entropy of an influence vector,
    based *only* on training images from the same class as the test image.
    """

    # 1. Create a boolean mask to find all training
    #    images that have the same label as the test image.
    in_class_mask = all_train_labels == test_label

    # 2. Handle the rare case where there are no
    #    training images of this class (e.g., bad data split).
    if not np.any(in_class_mask):
        return 0.0  # Return 0 entropy if no in-class images exist

    # 3. Filter the full influence vector to get only
    #    the scores from the in-class images.
    in_class_influence_vec = influence_vector[in_class_mask]

    # 4. Calculate entropy *only* on this filtered vector

    # Use absolute value for magnitude, add epsilon to prevent division by zero
    abs_influence = np.abs(in_class_influence_vec) + 1e-9

    # Normalize to create a probability distribution
    p_sum = np.sum(abs_influence)
    if p_sum == 0:
        return 0.0  # All in-class influences were zero

    p = abs_influence / p_sum

    # Calculate entropy
    # Add epsilon to log2 to prevent log(0)
    entropy = -np.sum(p * np.log2(p + 1e-9))

    return entropy


def wsl_path_to_windows(abs_img_path):
    """Converts a WSL /mnt/c/... path to a Windows C:/... path."""
    browser_path = abs_img_path
    if browser_path.startswith("/mnt/"):
        parts = browser_path.split("/", 3)
        if len(parts) == 4:
            drive_letter = parts[2].upper()
            rest_of_path = parts[3]
            browser_path = f"{drive_letter}:/{rest_of_path}"
    # Use 'file:///' for Windows paths (C:/...)
    return f"file:///{browser_path}"

def generate_influence_report(
    mem_scores,
    infl_scores,
    train_image_paths,
    train_labels,
    train_data_dir,
    test_image_paths,
    test_labels,
    test_data_dir,
    num_test_to_show=10,
    num_train_to_show=5,
    output_file='influence_report.html'
):
    """Generates a scrollable HTML report for influence scores."""

    n_test = infl_scores.shape[0]
    n_train = infl_scores.shape[1]

    print("Calculating IN-CLASS entropy for all test points...")
    
    # 1. Calculate entropy for all test points using the new method
    entropies = []
    for i in range(n_test):
        current_test_label = test_labels[i]
        current_influence_vec = infl_scores[i, :]
        
        e = calculate_entropy(
            current_influence_vec,
            current_test_label,
            train_labels
        )
        entropies.append(e)
    
    entropies = np.array(entropies)

    # 2. Get sorted indices based on entropy
    test_indices_sorted_by_entropy = np.argsort(entropies)
    
    # Get the two groups: lowest and highest entropy
    lowest_entropy_indices = test_indices_sorted_by_entropy[:num_test_to_show]
    highest_entropy_indices = test_indices_sorted_by_entropy[-num_test_to_show:][::-1] # Reverse to show highest first
    
    print(f"Generating report for Top-{num_test_to_show} lowest and highest entropy test points.")

    # 3. Start building HTML
    html_content = f"""
    <html>
    <head>
    <title>Influence Report (In-Class Entropy)</title>
    <style>
        /* ... (CSS style block is unchanged) ... */
        body {{ font-family: sans-serif; background-color: #f4f4f4; }}
        h1, h2 {{ text-align: center; }}
        .test-point-row {{
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            margin: 20px;
            padding: 10px;
            display: flex;
            align-items: center;
        }}
        .test-image-card {{
            flex: 0 0 200px; /* Fixed width for test image */
            text-align: center;
            padding: 10px;
            border-right: 2px solid #eee;
        }}
        .influencers-scroll-container {{
            display: flex;
            flex-direction: row;
            overflow-x: auto; /* This makes it scroll horizontally */
            flex: 1; /* Takes up remaining space */
            padding-left: 10px;
        }}
        .train-image-card {{
            flex: 0 0 180px; /* Fixed width for each train card */
            border: 1px solid #ccc;
            border-radius: 8px;
            margin: 5px;
            padding: 5px;
            text-align: center;
            background: #fafafa;
            overflow-wrap: break-word;
        }}
        .train-image-card.helpful {{ border-left: 5px solid green; }}
        .train-image-card.harmful {{ border-left: 5px solid red; }}
        
        img {{
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 4px;
        }}
        p {{ margin: 4px 0; font-size: 0.9em; }}
        .score {{ font-weight: bold; }}
        .mem-score {{ color: #00008B; }} /* Dark Blue */
        .infl-score-pos {{ color: #006400; }} /* Dark Green */
        .infl-score-neg {{ color: #8B0000; }} /* Dark Red */
    </style>
    </head>
    <body>
    <h1>Influence Report (In-Class Entropy)</h1>
    """

    # --- 4. Function to generate a section (Lowest or Highest) ---
    def create_section(title, test_indices):
        section_html = f"<h2>{title}</h2>"
        all_train_indices = np.arange(n_train) 
        
        for i, test_idx in enumerate(test_indices):
            
            # =======================================================
            # === ⬇️ EMBED THE TEST IMAGE ⬇️ ===
            # =======================================================
            test_img_name = test_image_paths[test_idx]
            test_img_abs_path = os.path.abspath(os.path.join(test_data_dir, test_img_name))
            test_img_src_data = embed_image_as_base64(test_img_abs_path)
            # =======================================================
            
            entropy = entropies[test_idx]
            test_label = test_labels[test_idx]
            
            # ... (Logic to find harmful/helpful indices is unchanged) ...
            influence_vec = infl_scores[test_idx, :]
            in_class_mask = (train_labels == test_label)
            in_class_train_indices = all_train_indices[in_class_mask]
            in_class_influence_vec = influence_vec[in_class_mask]
            in_class_sort_order = np.argsort(in_class_influence_vec)
            num_in_class = len(in_class_influence_vec)
            num_to_show = min(num_train_to_show, num_in_class)
            harmful_subset_indices = in_class_sort_order[:num_to_show]
            helpful_subset_indices = in_class_sort_order[-num_to_show:][::-1]
            harmful_indices = in_class_train_indices[harmful_subset_indices]
            helpful_indices = in_class_train_indices[helpful_subset_indices]

            section_html += f"""
            <div class="test-point-row">
                <div class="test-image-card">
                    <h3>Test Point (Rank {i+1})</h3>
                    <img src="{test_img_src_data}" alt="Test Image {test_idx}">
                    <p>Test Index: {test_idx}</p>
                    <p>Label: {test_label}</p>
                    <p class="score">In-Class Entropy: {entropy:.4f}</p>
                </div>
                <div class="influencers-scroll-container">
            """

            # Add helpful images
            for train_idx in helpful_indices:
                # =======================================================
                # === ⬇️ EMBED THE HELPFUL TRAIN IMAGE ⬇️ ===
                # =======================================================
                train_img_name = train_image_paths[train_idx]
                train_img_abs_path = os.path.abspath(os.path.join(train_data_dir, train_img_name))
                train_img_src_data = embed_image_as_base64(train_img_abs_path)
                # =======================================================
                
                infl = influence_vec[train_idx]
                mem = mem_scores[train_idx]
                
                section_html += f"""
                <div class="train-image-card helpful">
                    <img src="{train_img_src_data}" alt="Train {train_idx}" loading="lazy">
                    <p>Train Index: {train_idx}</p>
                    <p>Label: {train_labels[train_idx]}</p>
                    <p class="score infl-score-pos">Influence: {infl:+.4f}</p>
                    <p class="score mem-score">Memorization: {mem:.4f}</p>
                </div>
                """
            
            # Add harmful images
            for train_idx in harmful_indices:
                # =======================================================
                # === ⬇️ EMBED THE HARMFUL TRAIN IMAGE ⬇️ ===
                # =======================================================
                train_img_name = train_image_paths[train_idx]
                train_img_abs_path = os.path.abspath(os.path.join(train_data_dir, train_img_name))
                train_img_src_data = embed_image_as_base64(train_img_abs_path)
                # =======================================================

                infl = influence_vec[train_idx]
                mem = mem_scores[train_idx]
                
                section_html += f"""
                <div class="train-image-card harmful">
                    <img src="{train_img_src_data}" alt="Train {train_idx}" loading="lazy">
                    <p>Train Index: {train_idx}</p>
                    <p>Label: {train_labels[train_idx]}</p>
                    <p class="score infl-score-neg">Influence: {infl:+.4f}</p>
                    <p class="score mem-score">Memorization: {mem:.4f}</p>
                </div>
                """

            section_html += "</div></div>\n"
        return section_html
    # --- 5. Create both sections ---
    html_content += create_section("Lowest In-Class Entropy (Most Spiky) Test Points", lowest_entropy_indices)
    html_content += create_section("Highest In-Class Entropy (Most Uniform) Test Points", highest_entropy_indices)

    # --- 6. Finish and save ---
    html_content += "</body></html>"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nSuccess! Report generated: {output_file}")
    webbrowser.open(f'file://{os.path.abspath(output_file)}')


def main():
    parser = argparse.ArgumentParser(
        description="Generate a web-based influence report."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the 'AGGREGATED_RESULTS_FINAL.npz' file (must contain 'influence' and 'memorization').",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'cifar10'), to find the data dirs.",
    )
    parser.add_argument(
        "--num_test_points",
        type=int,
        default=10,
        help="Number of test points to show from the top and bottom of the entropy sort.",
    )
    parser.add_argument(
        "--num_train_points",
        type=int,
        default=5,
        help="Number of top helpful and harmful train images to show for each test point.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="influence_report.html",
        help="Name of the HTML file to generate.",
    )
    args = parser.parse_args()

    # --- 1. Load All Scores ---
    print(f"Loading scores from {args.results_file}...")
    try:
        data = np.load(args.results_file)
        mem_est = data["memorization"]
        infl_est = data["influence"]
    except Exception as e:
        print(f"Error loading results file: {e}")
        print(
            "Ensure 'AGGREGATED_RESULTS_FINAL.npz' contains 'memorization' and 'influence' arrays."
        )
        sys.exit(1)

    # --- 2. Load Both Train and Test Data Info ---
    print("Loading original train and test data paths/labels...")
    try:
        train_data_dir = os.path.join(
            "..", "data", args.dataset, "data", "img", "train"
        )
        test_data_dir = os.path.join("..", "data", args.dataset, "data", "img", "test")

        # This function returns test *users*, but we want the flat test_data
        # Your read_data seems to return test_data["100"] as the key
        # We need to load *both* test and train data
        _, train_data_dict, test_data_dict = read_data(
            train_data_dir, test_data_dir, alpha=0
        )

        train_image_paths = train_data_dict["x"]
        train_labels = train_data_dict["y"]

        # This assumes your flat test set is under the key "100" as in your main script
        test_image_paths = test_data_dict["100"]["x"]
        test_labels = test_data_dict["100"]["y"]

        # HACK: If '100' isn't the key, try to find another.
        # This is brittle and depends on your 'read_data' structure.
        if not isinstance(test_image_paths, np.ndarray):
            print(
                "Warning: test_data_dict['100'] not found. Trying first available test key."
            )
            first_test_key = list(test_data_dict.keys())[0]
            test_image_paths = test_data_dict[first_test_key]["x"]
            test_labels = test_data_dict[first_test_key]["y"]

    except Exception as e:
        print(f"Error loading training data with read_data: {e}")
        print(
            "Please ensure 'utils.model_utils' is importable and your data is in the correct path."
        )
        sys.exit(1)

    # --- 3. Validation ---
    if infl_est.shape[1] != len(train_image_paths):
        print("Error: Influence matrix train dimension mismatch!")
        print(f"Influence shape[1]: {infl_est.shape[1]}")
        print(f"Training images found: {len(train_image_paths)}")
        sys.exit(1)

    if infl_est.shape[0] != len(test_image_paths):
        print("Error: Influence matrix test dimension mismatch!")
        print(f"Influence shape[0]: {infl_est.shape[0]}")
        print(f"Test images found: {len(test_image_paths)}")
        sys.exit(1)

    if mem_est.shape[0] != len(train_image_paths):
        print("Error: Memorization array dimension mismatch!")
        print(f"Memorization shape[0]: {mem_est.shape[0]}")
        print(f"Training images found: {len(train_image_paths)}")
        sys.exit(1)

    # --- 4. Generate Report ---
    generate_influence_report(
        mem_scores=mem_est,
        infl_scores=infl_est,
        train_image_paths=train_image_paths,
        train_labels=train_labels,
        train_data_dir=train_data_dir,
        test_image_paths=test_image_paths,
        test_labels=test_labels,
        test_data_dir=test_data_dir,  # <-- Pass the test data dir
        num_test_to_show=args.num_test_points,
        num_train_to_show=args.num_train_points,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
