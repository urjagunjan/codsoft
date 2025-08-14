# codsoft
This directory contains all the tasks, projects, and learning resources completed during my AI-focused internship at CodSoft.  
# Image Captioning AI with SQLite Integration

This project implements an automatic image captioning system using deep learning. It combines a pre-trained VGG16 model for image feature extraction with an LSTM-based sequence model for generating descriptive captions. The project now features an SQLite database backend for efficient data management.

## ✨ Features

- **Efficient Data Management**: Uses SQLite database to store image features and captions
- **Deep Learning Pipeline**: Combines VGG16 for image features with LSTM for sequence generation
- **Easy Setup**: Simple configuration and setup process
- **Flexible**: Can be extended with different datasets or models

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ImageCaptioningAI.git
   cd ImageCaptioningAI
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Setup

1. **Download the Flickr8k dataset** from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)
   - Download both `Flickr8k_text.zip` and `Flickr_8k_Images.zip`

2. **Extract the files** and organize them as follows:
   ```
   ImageCaptioningAI/
   ├── Flicker8k_Dataset/    # All images go here
   ├── captions.txt          # From Flickr8k_text.zip
   ├── Flickr_8k.testImages.txt
   ├── Flickr_8k.trainImages.txt
   └── caption.py
   ```

### Database Initialization

The first time you run the script, it will automatically:
1. Create an SQLite database (`captions.db`)
2. Process and store image features
3. Store all captions in the database

## 🏃‍♂️ Usage

### Training the Model
```bash
python caption.py --mode train --epochs 20 --batch_size 32
```

### Generating Captions
```bash
python caption.py --mode predict --image_path path/to/your/image.jpg
```

### Available Arguments
- `--mode`: Operation mode - 'train' or 'predict' (default: 'train')
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 32)
- `--image_path`: Path to image for caption generation (required for predict mode)

## 🏗️ Project Structure

- `caption.py`: Main script containing the complete pipeline
- `requirements.txt`: Python dependencies
- `captions.db`: SQLite database (created automatically)
- `.gitignore`: Specifies intentionally untracked files to ignore

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
