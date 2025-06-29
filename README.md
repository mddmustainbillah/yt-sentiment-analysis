# YouTube Sentiment Analysis 🚀

<!-- Replace YOUR_VIDEO_ID with your actual YouTube video ID -->
<!-- The video ID is the part after "v=" in your YouTube URL -->
<!-- Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ -> Video ID is "dQw4w9WgXcQ" -->

## 🎥 Project Demo

[![YouTube Sentiment Analysis Demo](https://img.youtube.com/vi/YVju3f0Rd2s/maxresdefault.jpg)](https://www.youtube.com/watch?v=YVju3f0Rd2s)

*Click the image above to watch the project demonstration*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Chrome Extension](#chrome-extension)
- [Future Work](#future-work)


---

## 🌟 Overview

Have you ever tried to follow a technical tutorial on YouTube, like "how to install Windows on a Mac," only to find out after wasting hours that the instructions don't work? I have, and it's incredibly frustrating. That's why I built this project.

This tool analyzes the sentiment of YouTube comments to give you a quick idea of whether a video is helpful and if the solution it presents actually works. The project features a machine learning model that classifies comments as positive, negative, or neutral, and a companion Chrome extension for real-time sentiment analysis directly on the YouTube platform.

---

## 🎯 Objectives

- **💾 Save Time**: Help users quickly determine if a video is worth watching based on the collective sentiment of the comments
- **📊 Analyze Viewer Sentiment**: Understand the overall sentiment of viewers towards a YouTube video
- **📈 Identify Trends**: Detect patterns and trends in viewer comments
- **⚡ Provide Real-time Feedback**: Offer content creators instant feedback on their videos

---

## ✨ Features

- 🎯 **Sentiment Analysis**: Classifies comments into positive, negative, and neutral categories
- ⚡ **Real-time Analysis**: Chrome extension that displays sentiment analysis directly on YouTube video pages
- 📊 **MLflow Integration**: Tracks experiments, logs model performance, and manages the model lifecycle
- 🔄 **DVC for Data Versioning**: Manages large data files and ensures reproducibility
- 🚀 **CI/CD Pipeline**: Automates testing and deployment using GitHub Actions
- 🌐 **Flask API**: Simple web API to serve the model and expose prediction endpoints
- 🐳 **Dockerized Application**: Containerized with Docker for easy deployment and scalability

---

## 🛠️ Tech Stack

### Backend & ML
- **Python** - Core programming language
- **Flask** - Web framework for API
- **Scikit-learn** - Machine learning library
- **NLTK** - Natural language processing
- **LightGBM** - Gradient boosting framework

### MLOps & DevOps
- **MLflow** - ML lifecycle management
- **DVC** - Data version control
- **Docker** - Containerization
- **GitHub Actions** - CI/CD pipeline

### Environment Management
- **virtualenv** - Python environment management

---

## 📂 Project Structure

```
├── data/                 # Raw, processed, and interim data
├── flask_app/           # Flask application for serving the model
├── models/              # Trained machine learning models
├── notebooks/           # Jupyter notebooks for experimentation
├── src/                 # Source code for data processing, model training, etc.
├── .github/workflows/   # CI/CD pipeline configuration
├── Dockerfile           # Docker configuration
├── dvc.yaml            # DVC pipeline definition
├── mlruns/             # MLflow experiment tracking data
├── params.yaml         # Parameters for the project
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Git
- Docker (optional)
- DVC (for pipeline management)

### Step-by-step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/yt-sentiment-analysis.git
   cd yt-sentiment-analysis
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the DVC pipeline**
   ```bash
   dvc repro
   ```

---

## 📖 Usage

### Running the ML Pipeline

To train the model and process data:

```bash
dvc repro
```

This command will run the entire machine learning pipeline as defined in `dvc.yaml`, including data processing, model training, and evaluation.

### Running the Flask Application

1. **Navigate to the Flask app directory**
   ```bash
   cd flask_app
   ```

2. **Start the application**
   ```bash
   python app.py
   ```

3. **Access the application**
   - Open your browser and go to: `http://127.0.0.1:5000`

### Running with Docker

1. **Build the Docker image**
   ```bash
   docker build -t yt-sentiment-analysis .
   ```

2. **Run the Docker container**
   ```bash
   docker run -p 5000:5000 yt-sentiment-analysis
   ```

3. **Access the application**
   - Open your browser and go to: `http://127.0.0.1:5000`

---

## 🌐 Chrome Extension

The Chrome extension provides a user-friendly interface to see the sentiment analysis in real-time on YouTube.

### Installation
- **Extension Repository**: [Chrome Extension Repository](https://github.com/your-username/your-chrome-extension-repo)
- Follow the installation instructions in the extension repository

### Features
- Real-time sentiment analysis on YouTube videos
- Visual indicators for comment sentiment
- Quick overview of overall video sentiment


---

## 🔮 Future Work

- 🌍 **Multi-language Support**: Extend the model to analyze comments in languages other than English
- 🎯 **Aspect-Based Sentiment**: Provide granular insights by identifying sentiment towards specific aspects (e.g., "audio quality", "explanation clarity")
- 📊 **Advanced Visualizations**: Enhance the Chrome extension with interactive and insightful visualizations
- 👤 **User Authentication**: Allow users to save analysis history and preferences
- 🤖 **Real-time Model Updates**: Implement continuous learning capabilities

---



**⭐ If you found this project helpful, please give it a star!**