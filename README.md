# American Sign Language (ASL) Detection Using CNN and LSTM

## Project Overview
This project implements an American Sign Language (ASL) detection system using a Convolutional Neural Network (CNN) and Long Short-Term Memory (LSTM) networks. The backend is built using Flask, which provides a REST API for handling live video input and returning predictions for ASL signs. The system is designed to interpret hand gestures from images and classify them into corresponding letters of the ASL alphabet.

## Project Structure
```
├── app.py                # Main Flask application
├── checkpoint
│   └── model.tflite      # Pre-trained CNN-LSTM model
├── static                # Folder to store uploaded images
├── templates
│   └── index.html        # Home page
├── utils.py              # Helper functions for preprocessing input images
├── README.md             # Project documentation
```

## Features
- **Prediction**: Returns the corresponding ASL character for the frame.
- **Web Interface**: A simple web interface for users to interact with the system.
- **REST API**: Provides endpoints for automated integration with other systems.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GShankar555/American-Sign-Language-Detection.git
   cd ASL
   ```

2. **Download Dataset:**
   ```bash
   https://drive.google.com/file/d/1Epb3M1nAIIku185Fk8tLq2nQwn7X8LSS/view?usp=sharing
   ```

3. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

## Usage

1. **Run the Flask server:**
   ```bash
   python app.py
   ```

2. **Access the web interface:**
   Open your browser and navigate to `http://127.0.0.1:5000`.

## Dependencies
The following libraries are required to run this project:
- Flask
- TensorFlow
- Keras
- OpenCV
- NumPy

---
