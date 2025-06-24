### Environment Versions

* **Python**: `3.12.6`
* **PyTorch**: `2.4.0`

### Project Structure

* **`model.py`**: This file contains the architecture implementation for the core **CAMFT** model.

* **`log/`**: This directory stores the experimental logs for all models.
    * Logs are organized into subdirectories by model name (e.g., `CAMFT`, `EEGNet`) and ablation studies (`CAMFT_ablation`).
    * Within a specific model's directory (e.g., `log/CAMFT/`), results are further archived by learning rate (e.g., `1e-05`).

* **`log_significance_analysis/`**: This directory holds the statistical significance test results comparing CAMFT against other models.
