# 🔍 AI Glaucoma Detection System

An advanced web-based glaucoma detection system powered by DenseNet121 deep learning model. This application analyzes retinal fundus images to detect signs of glaucoma with high accuracy, providing instant results and professional PDF reports.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.12+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌟 Features

- **🤖 AI-Powered Detection**: Uses DenseNet121 deep learning model for accurate glaucoma detection
- **📊 Real-time Analysis**: Instant image processing and results in under 5 seconds
- **💻 Modern Web Interface**: Beautiful, responsive UI with confidence level visualization
- **📄 Professional Reports**: Generate detailed PDF reports with medical recommendations
- **🛡️ Production Ready**: Memory optimization, auto-cleanup, and crash prevention
- **📱 Mobile Friendly**: Responsive design works on all devices

## 🎯 Model Performance

- **Architecture**: DenseNet121 (121-layer densely connected CNN)
- **Image Size**: 224×224 pixels
- **Classes**: Normal, Glaucoma
- **Accuracy**: High accuracy validated on medical datasets

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mk-pavel/Glaucoma_Detection.git
   cd Glaucoma_Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your trained model**
   - Place your `DenseNet121_glaucoma.h5` model file in the `models/` directory

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://127.0.0.1:5000`

## 📁 Project Structure

```
Glaucoma_Detection/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── models/               # Model directory
│   └── DenseNet121_glaucoma.h5  # Your trained model
├── static/               # Static assets
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── script.js
│   └── images/
│       ├── hero-bg.jpg
│       ├── about.jpg
│       ├── technology.jpg
│       └── favicon.png
├── templates/            # HTML templates
│   ├── base.html
│   ├── index.html
│   └── result.html
└── uploads/              # Temporary file uploads
```

## 🔧 Usage

### Web Interface

1. **Upload Image**: Drag & drop or browse to select a retinal fundus image
2. **Analyze**: Click "Analyze Image" to process the image
3. **View Results**: See prediction results with confidence levels
4. **Download Report**: Generate and download a professional PDF report

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## 🛠️ Configuration

### Memory Management

The application includes automatic memory management:
- Auto-cleanup of old files (>1 hour)
- TensorFlow GPU memory growth configuration
- Matplotlib memory leak prevention
- Garbage collection optimization

### File Size Limits

- Maximum upload size: 16MB
- Recommended image size: 224×224 to 1024×1024 pixels

## 📊 API Endpoints

- `GET /` - Main application interface
- `POST /` - Image upload and analysis
- `GET /uploads/<filename>` - Serve uploaded images
- `POST /download_report` - Generate and download PDF report

## 🔬 Medical Disclaimer

⚠️ **IMPORTANT**: This AI system is designed for screening and educational purposes only and should not replace professional medical diagnosis. Always consult with a qualified ophthalmologist or optometrist for proper medical evaluation.

## 🛡️ Security Features

- Secure filename handling
- File type validation
- Size limit enforcement
- Auto-cleanup of temporary files
- Error handling and logging

## 📈 Performance Optimization

- Non-blocking matplotlib backend
- Memory-efficient image processing
- Automatic resource cleanup
- Threaded Flask server
- Production-ready configuration

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**MK Pavel**
- GitHub: [@mk-pavel](https://github.com/mk-pavel)

## 🙏 Acknowledgments

- DenseNet architecture by Huang et al.
- TensorFlow and Keras teams
- Flask web framework
- Bootstrap for UI components
- Medical datasets used for training

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/mk-pavel/Glaucoma_Detection/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

---

⭐ **Star this repository if you find it useful!**
