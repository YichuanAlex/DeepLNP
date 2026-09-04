"""Download datasets for DeepLNP."""

import os
import requests
from pathlib import Path
from typing import Optional


def download_dataset(
    dataset_name: str,
    output_dir: str = "data/raw",
    url: Optional[str] = None
) -> Path:
    """
    Download a dataset from a URL or local path.
    
    Args:
        dataset_name: Name of the dataset
        output_dir: Directory to save the dataset
        url: URL to download from (optional)
    
    Returns:
        Path to downloaded file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_name == "lnp_atlas":
        # LNP Atlas dataset
        if url is None:
            # Placeholder: replace with actual URL from the paper
            print("Please download LNP Atlas from the original repository")
            print("https://github.com/.../LNP_Atlas")
            return None
        
        filename = "lnp_atlas.csv"
    elif dataset_name == "agile":
        # AGILE dataset
        if url is None:
            print("Please download AGILE from the original repository")
            print("https://github.com/bowang-lab/AGILE")
            return None
        
        filename = "agile.csv"
    elif dataset_name == "lantern":
        # LANTERN dataset
        if url is None:
            print("Please download LANTERN from the original repository")
            print("https://github.com/AsalMehradfar/LANTERN")
            return None
        
        filename = "lantern_cleaned.csv"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    output_path = output_dir / filename
    
    if output_path.exists():
        print(f"Dataset already exists: {output_path}")
        return output_path
    
    if url:
        print(f"Downloading {dataset_name} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded to {output_path}")
    else:
        print(f"Manual download required for {dataset_name}")
        print(f"Save the file to: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download datasets for DeepLNP")
    parser.add_argument(
        "--dataset",
        type=str,
        default="lnp_atlas",
        choices=["lnp_atlas", "agile", "lantern"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw",
        help="Output directory"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL to download from"
    )
    
    args = parser.parse_args()
    
    download_dataset(args.dataset, args.output, args.url)
