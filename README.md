# Adapted Speech Separation Models for 1D Signals

This repository periodically releases **adapted versions of existing speech separation models**, tailored for **multi-channel 1D signal processing** (e.g., EEG, ECG, sensor time-series, etc.).

## 📌 Key Features

- **Equal input and output length**: All adapted models preserve temporal dimensionality — input and output sequences have the same length, making them suitable for private datasets.
- **Based on existing models**: Each implementation is derived from publicly available source code. Original sources are explicitly cited in the corresponding files.
- **Ongoing updates**: New adapted models will be added intermittently.

## ⚠️ Disclaimer

The performance of these adapted models has **not been rigorously validated**. Use at your own discretion.  
If you find any errors or inconsistencies, please feel free to open an issue or contact me directly.

## 🛠️ Usage

- Models are modified to accept generic multi-channel 1D signals as input.
- The equal-length I/O design ensures compatibility with downstream tasks requiring strict time alignment (e.g., residual computation, echo cancellation).
- Attribution to original authors is included in code comments and/or license headers.

## 📄 License

This code is intended for research and educational purposes only. Please refer to individual files for original model licenses and attributions.
