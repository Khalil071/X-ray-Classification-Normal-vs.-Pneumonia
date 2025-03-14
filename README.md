X-ray Classification: Normal vs. Pneumonia

Overview

This project implements and compares Convolutional Neural Networks (CNN), MobileNetV2, and ResNet50 models for detecting and classifying X-ray images into two categories:

Normal

Pneumonia

The goal is to develop an accurate deep-learning model to assist in diagnosing pneumonia from chest X-ray images.

Dataset

The dataset used is xray_dataset_covid19, which contains labeled X-ray images of normal and pneumonia cases. The dataset is structured as follows:

├── train/
│   ├── NORMAL/
│   ├── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   ├── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   ├── PNEUMONIA/

Project Structure

├── dataset/                     # Contains X-ray dataset
├── models/                      # Saved trained models
├── notebooks/                   # Jupyter notebooks for model training and evaluation
├── src/                         # Source code
│   ├── cnn_model.py             # CNN model implementation
│   ├── mobilenet_model.py       # MobileNetV2 model implementation
│   ├── resnet_model.py          # ResNet50 model implementation
│   ├── train.py                 # Training script for all models
│   ├── evaluate.py              # Evaluation script
│   ├── preprocess.py            # Preprocessing functions for dataset
├── requirements.txt             # Required dependencies
├── README.md                    # Project documentation

Installation

Clone the repository:

git clone https://github.com/your-username/xray-classification.git
cd xray-classification

Create a virtual environment (optional but recommended):

python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Model Training

Run the training script to train the models:

python src/train.py --model cnn        # Train CNN model
python src/train.py --model mobilenet  # Train MobileNetV2 model
python src/train.py --model resnet     # Train ResNet50 model

Evaluation

Evaluate the trained models on the test set:

python src/evaluate.py --model cnn        # Evaluate CNN model
python src/evaluate.py --model mobilenet  # Evaluate MobileNetV2 model
python src/evaluate.py --model resnet     # Evaluate ResNet50 model

Results

The models are compared based on accuracy, precision, recall, and F1-score. Detailed analysis and visualizations are available in the Jupyter notebooks under notebooks/.

Future Improvements

Implementing other deep-learning architectures such as EfficientNet.

Hyperparameter tuning for improved accuracy.

Data augmentation to enhance model generalization.

Deployment of the best model as a web-based diagnostic tool.

Contributing

Feel free to contribute by opening issues or submitting pull requests!

License

This project is licensed under the MIT License.
