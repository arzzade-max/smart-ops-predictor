"""
Data generator for smart operations predictor.

Generates synthetic operational data for testing and training the SLA breach prediction model.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_data_path(filename: str = "ops_data.csv") -> str:
    """
    Get the absolute path to the data directory.
    
    Args:
        filename: Name of the file to save (default: "ops_data.csv")
    
    Returns:
        str: Absolute path to the file in the data directory
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    return str(data_dir / filename)


def generate_data(n: int = 2000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic operational data.
    
    Args:
        n: Number of records to generate (default: 2000)
        random_seed: Random seed for reproducibility (default: 42)
    
    Returns:
        pd.DataFrame: Generated dataset with columns for ticket operations
    """
    np.random.seed(random_seed)
    
    data = pd.DataFrame({
        "ticket_id": range(1, n + 1),
        "handle_time": np.random.randint(5, 60, n),
        "priority": np.random.choice([1, 2, 3], n),
        "agent_experience": np.random.randint(1, 10, n),
        "queue_load": np.random.randint(50, 300, n)
    })

    # Calculate SLA breach based on business rules
    data["SLA_breached"] = (
        (data["handle_time"] > 40) |
        (data["queue_load"] > 200) |
        ((data["priority"] == 1) & (data["agent_experience"] < 3))
    ).astype(int)

    return data


def save_data(data: pd.DataFrame, output_path: str) -> None:
    """
    Save data to CSV file with proper error handling.
    
    Args:
        data: DataFrame to save
        output_path: Full path to the output file
    
    Raises:
        IOError: If unable to write to the specified path
    """
    try:
        # Create parent directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        data.to_csv(output_path, index=False)
        logger.info(f"Dataset saved successfully to {output_path}")
        logger.info(f"Records created: {len(data)}")
        
    except PermissionError:
        logger.error(f"Permission denied: Unable to write to {output_path}")
        raise IOError(f"Permission denied when writing to {output_path}") from None
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")
        raise IOError(f"Failed to save dataset to {output_path}: {e}") from None


def main() -> int:
    """
    Main entry point for the data generation script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        logger.info("Starting data generation...")
        
        # Generate data
        data = generate_data()
        
        # Get output path and save
        output_path = get_data_path()
        save_data(data, output_path)
        
        logger.info("Data generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Data generation failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())