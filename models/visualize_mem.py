import numpy as np
import os
import argparse
import webbrowser
from utils.model_utils import read_data  # <-- Assumes this import works
import sys
import base64
from PIL import Image
import io


def image_to_base64(image_path):
    """Convert an image file to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save to bytes buffer
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            img_bytes = buffer.getvalue()

            # Encode to base64
            base64_str = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return None


def generate_html_report(
    mem_scores,
    image_paths,
    labels,
    train_data_dir,  # <-- CHANGED: Added train_data_dir
    sort_order="descending",
    top_k=500,
    output_file="memorization_report.html",
):
    """Generates a scrollable HTML report of images and their scores."""

    print(f"Generating report with Top-{top_k} images, sorted {sort_order}...")

    # 1. Get sorted indices
    if sort_order == "descending":
        sorted_indices = np.argsort(mem_scores)[::-1]
    else:
        sorted_indices = np.argsort(mem_scores)

    # 2. Filter for Top-K
    top_k_indices = sorted_indices[:top_k]

    # 3. Start building the HTML
    html_content = f"""
    <html>
    <head>
    <title>Memorization Report</title>
    <style>
        body {{ font-family: sans-serif; background-color: #f4f4f4; }}
        h1 {{ text-align: center; }}
        .container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 20px;
            padding: 20px;
        }}
        .item {{
            border: 1px solid #ccc;
            border-radius: 8px;
            background: #fff;
            box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
            text-align: center;
            padding-bottom: 10px;
            overflow-wrap: break-word;
        }}
        .item img {{
            width: 100%;
            height: 150px; /* Fixed height for consistency */
            object-fit: cover;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }}
        .item p {{ margin: 5px 0; }}
        .score {{ font-weight: bold; font-size: 1.1em; }}
    </style>
    </head>
    <body>
    <h1>Memorization Report</h1>
    <h2>Showing Top {top_k} Samples (Sorted {sort_order})</h2>
    <div class="container">
    """

    # 4. Loop through and create an HTML block for each image
    print("Converting images to base64...")
    for i, idx in enumerate(top_k_indices):
        score = mem_scores[idx]
        img_name = image_paths[idx]
        label = labels[idx]

        # Construct the full path
        full_img_path = os.path.join(train_data_dir, img_name)

        # Convert image to base64
        base64_img = image_to_base64(full_img_path)

        if base64_img:
            img_tag = f'<img src="{base64_img}" alt="Image {idx}">'
        else:
            img_tag = '<div class="error">Image not found</div>'

        html_content += f"""
        <div class="item">
            {img_tag}
            <p class="score">Score: {score:.4f}</p>
            <p>Rank: {i + 1}</p>
            <p>Train Index: {idx}</p>
            <p>Label: {label}</p>
        </div>
        """

    # 5. Close the HTML tags
    html_content += """
    </div>
    </body>
    </html>
    """

    # 6. Write to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nSuccess! Report generated: {output_file}")

    # 7. Auto-open in browser
    webbrowser.open(f"file://{os.path.abspath(output_file)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a web-based memorization report."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the 'AGGREGATED_RESULTS_FINAL.npz' file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset (e.g., 'cifar10'), to find the data dir.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=500,
        help="Number of images to include in the report.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="descending",
        choices=["descending", "ascending"],
        help="Sort order (descending = most memorized first).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="memorization_report.html",
        help="Name of the HTML file to generate.",
    )
    args = parser.parse_args()

    # --- 1. Load Memorization Scores ---
    print(f"Loading scores from {args.results_file}...")
    try:
        data = np.load(args.results_file)
        mem_est = data["memorization"]
    except Exception as e:
        print(f"Error loading results file: {e}")
        sys.exit(1)

    # --- 2. Load Original Training Data Info ---
    print("Loading original training data paths and labels...")
    try:
        # Construct paths as in your main script
        train_data_dir = os.path.join(
            "..", "data", args.dataset, "data", "img", "train"
        )
        test_data_dir = os.path.join("..", "data", args.dataset, "data", "img", "test")

        _, train_data, _ = read_data(train_data_dir, test_data_dir, alpha=0)

        image_paths = train_data["x"]  # This is just the image *names*
        labels = train_data["y"]
    except Exception as e:
        print(f"Error loading training data with read_data: {e}")
        print(
            "Please ensure 'utils.model_utils' is importable and your data is in the correct path."
        )
        sys.exit(1)

    # --- 3. Validation ---
    if len(mem_est) != len(image_paths):
        print("Error: Mismatch in data sizes!")
        print(f"Memorization scores: {len(mem_est)}")
        print(f"Training images found: {len(image_paths)}")
        print(
            "Make sure you are using the same dataset and the 'sorted()' fix is in your read_data function."
        )
        sys.exit(1)

    # --- 4. Generate Report ---
    generate_html_report(
        mem_scores=mem_est,
        image_paths=image_paths,
        labels=labels,
        train_data_dir=train_data_dir,  # <-- CHANGED: Pass the dir path
        sort_order=args.sort,
        top_k=args.top_k,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
