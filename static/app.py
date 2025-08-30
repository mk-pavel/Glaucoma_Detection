import os
import io
import gc
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
from werkzeug.utils import secure_filename
import tensorflow as tf

# Configure TensorFlow memory growth to prevent OOM
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -------------------------------
# Flask Setup
# -------------------------------
app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "models/DenseNet121_glaucoma.h5"   # <-- use your saved DenseNet121 model
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained DenseNet121 model
model = load_model(MODEL_PATH)
IMG_SIZE = 224   # use same size as training (you used 224 for DenseNet121)

# Class labels (for plotting only)
class_labels = ["Normal", "Glaucoma"]

# -------------------------------
# Cleanup Functions
# -------------------------------
def cleanup_old_files():
    """Clean up old files to prevent disk space issues"""
    try:
        upload_dir = app.config["UPLOAD_FOLDER"]
        current_time = time.time()
        
        # Remove files older than 1 hour
        for filename in os.listdir(upload_dir):
            filepath = os.path.join(upload_dir, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                # Delete files older than 1 hour (3600 seconds)
                if file_age > 3600:
                    try:
                        os.remove(filepath)
                        print(f"Cleaned up old file: {filename}")
                    except OSError:
                        pass  # File might be in use
    except Exception as e:
        print(f"Cleanup error: {e}")


# -------------------------------
# Prediction Function
# -------------------------------
def model_predict(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Use same logic as your Colab code
        pred = model.predict(img_array)[0][0]  # Single probability value
        
        # Apply same threshold logic as Colab
        if pred > 0.5:
            prediction = "Glaucoma"
            confidence = pred * 100  # Confidence as percentage
        else:
            prediction = "Normal" 
            confidence = (1 - pred) * 100  # Confidence for Normal class
        
        # Create preds array for plotting (showing both probabilities)
        preds = np.array([1-pred, pred])  # [Normal_prob, Glaucoma_prob]
        
        # Clean up variables to prevent memory leaks
        del img, img_array
        gc.collect()
        
        return prediction, confidence, preds
        
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return "Error", 0.0, np.array([0.5, 0.5])


# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        if file:
            # Clean up old files periodically
            cleanup_old_files()
            
            # Create unique filename with timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                # Prediction
                pred_class, confidence, preds = model_predict(filepath, model)
                
                # Plot confidence chart with proper cleanup
                plt.ioff()  # Turn off interactive mode
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.bar(class_labels, preds * 100, color=["#27ae60", "#e74c3c"])
                ax.set_title("Prediction Confidence", fontsize=14, pad=20)
                ax.set_ylabel("Confidence (%)", fontsize=12)
                ax.set_ylim(0, 100)
                
                # Add percentage labels on bars
                for i, v in enumerate(preds * 100):
                    ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plot_path = os.path.join(app.config["UPLOAD_FOLDER"], f"plot_{timestamp}.png")
                plt.savefig(plot_path, dpi=100, bbox_inches='tight')
                plt.close(fig)  # Close specific figure
                plt.clf()       # Clear the figure
                plt.cla()       # Clear the axes
                
                # Force garbage collection
                gc.collect()

                # Create result object for template
                result = {
                    'prediction': pred_class,
                    'confidence': round(confidence, 2),
                    'filename': filename
                }
                
                return render_template("result.html", result=result)
                
            except Exception as e:
                print(f"Error processing image: {e}")
                # Clean up the uploaded file if processing failed
                try:
                    os.remove(filepath)
                except:
                    pass
                return render_template("index.html", error="Error processing image. Please try again.")

    return render_template("index.html")


# -------------------------------
# Serve uploaded files
# -------------------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))


# -------------------------------
# PDF Report Route
# -------------------------------
@app.route("/download_report", methods=["POST"])
def download_report():
    prediction = request.form.get("prediction")
    confidence = request.form.get("confidence")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Get styles
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = styles['Title']
    title_style.fontSize = 24
    title_style.spaceAfter = 0.3*inch
    title_style.textColor = colors.HexColor('#2c3e50')
    
    heading_style = styles['Heading2']
    heading_style.fontSize = 16
    heading_style.textColor = colors.HexColor('#34495e')
    heading_style.spaceAfter = 0.2*inch
    
    # Header with logo effect
    title = Paragraph("AI GLAUCOMA DETECTION REPORT", title_style)
    story.append(title)
    story.append(Spacer(1, 0.2*inch))
    
    # Information Table
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    info_data = [
        ['Analysis Date:', current_time],
        ['AI Model:', 'DenseNet121 (Deep Learning)'],
        ['Image Resolution:', '224×224 pixels'],
        ['Processing Time:', '< 5 seconds']
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Results Section
    results_heading = Paragraph("ANALYSIS RESULTS", heading_style)
    story.append(results_heading)
    
    # Determine colors and status based on prediction
    if prediction == "Glaucoma":
        result_color = colors.HexColor('#e74c3c')
        confidence_color = colors.HexColor('#c0392b')
        status = "Signs of Glaucoma Detected"
    else:
        result_color = colors.HexColor('#27ae60')
        confidence_color = colors.HexColor('#229954')
        status = "No Signs of Glaucoma"
    
    # Results table
    results_data = [
        ['Prediction:', status],
        ['Confidence Level:', f"{confidence}%"],
        ['Risk Assessment:', 'High' if prediction == 'Glaucoma' else 'Low']
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 3*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (1, 0), (1, 0), result_color),  # Prediction row
        ('TEXTCOLOR', (1, 1), (1, 1), confidence_color),  # Confidence row
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('FONTSIZE', (1, 0), (1, 0), 14),  # Make prediction larger
        ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Recommendations Section
    recommendations_heading = Paragraph("MEDICAL RECOMMENDATIONS", heading_style)
    story.append(recommendations_heading)
    
    if prediction == "Glaucoma":
        # Important notice
        important_text = "<para fontSize='12' textColor='#e74c3c' leftIndent='20' rightIndent='20' spaceAfter='12'><b>IMPORTANT:</b> Please consult with an ophthalmologist immediately for comprehensive evaluation.</para>"
        story.append(Paragraph(important_text, styles['BodyText']))
        
        # Recommendations list
        rec_data = [
            ['•', 'Schedule an urgent appointment with an eye specialist'],
            ['•', 'Early detection and treatment are crucial for preventing vision loss'],
            ['•', 'Bring this report to your ophthalmologist appointment'],
            ['•', 'Consider additional diagnostic tests (OCT, visual field testing)']
        ]
        
    else:
        # Good news notice  
        good_news_text = "<para fontSize='12' textColor='#27ae60' leftIndent='20' rightIndent='20' spaceAfter='12'><b>GOOD NEWS:</b> No signs of glaucoma detected in this analysis.</para>"
        story.append(Paragraph(good_news_text, styles['BodyText']))
        
        # Recommendations list
        rec_data = [
            ['•', 'Continue regular eye examinations as recommended by your doctor'],
            ['•', 'Maintain a healthy lifestyle and protect your eyes'],
            ['•', 'Schedule routine check-ups every 1-2 years'],
            ['•', 'Report any sudden vision changes to your doctor immediately']
        ]
    
    # Create recommendations table
    rec_table = Table(rec_data, colWidths=[0.3*inch, 4.5*inch])
    rec_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#2c3e50')),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    story.append(rec_table)
    story.append(Spacer(1, 0.4*inch))
    
    # Disclaimer Section
    disclaimer_style = styles['BodyText']
    disclaimer_style.fontSize = 9
    disclaimer_style.textColor = colors.HexColor('#7f8c8d')
    disclaimer_style.leftIndent = 20
    disclaimer_style.rightIndent = 20
    
    # Split disclaimer into multiple paragraphs for better parsing
    disclaimer_title = "<para fontSize='10' textColor='#7f8c8d'><b>IMPORTANT DISCLAIMER:</b></para>"
    story.append(Paragraph(disclaimer_title, disclaimer_style))
    
    disclaimer_main = """This AI analysis is designed for screening and educational purposes only and should not replace professional medical diagnosis. The results are based on artificial intelligence analysis of retinal images and should be interpreted by qualified healthcare professionals. Please consult with a licensed ophthalmologist or optometrist for proper medical evaluation, diagnosis, and treatment recommendations."""
    
    disclaimer_paragraph = "<para fontSize='9' textColor='#7f8c8d' leftIndent='20' rightIndent='20' spaceAfter='10'>" + disclaimer_main + "</para>"
    story.append(Paragraph(disclaimer_paragraph, disclaimer_style))
    
    # Report info
    report_id = datetime.now().strftime('%Y%m%d%H%M%S')
    report_info = f"<para fontSize='9' textColor='#7f8c8d' leftIndent='20' rightIndent='20'><b>Report generated by:</b> DenseNet121 AI Model | <b>Report ID:</b> GLU-{report_id}</para>"
    story.append(Paragraph(report_info, disclaimer_style))
    
    # Build PDF
    try:
        doc.build(story)
        buffer.seek(0)
        
        # Create a copy of the buffer to avoid issues with cleanup
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Create new buffer for response
        response_buffer = io.BytesIO(pdf_data)
        
        # Force garbage collection
        gc.collect()
        
        return send_file(response_buffer, as_attachment=True, download_name="AI_Glaucoma_Detection_Report.pdf", mimetype="application/pdf")
    
    except Exception as e:
        print(f"Error generating PDF: {e}")
        buffer.close()
        return render_template("result.html", error="Error generating PDF report. Please try again.")


# -------------------------------
# Run Flask
# -------------------------------
if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    # Set flask configuration for better memory management
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
    
    print("🚀 Starting AI Glaucoma Detection Server...")
    print("📊 Model loaded successfully")
    print("🔧 Memory optimization enabled")
    print("🧹 Auto cleanup configured")
    print("✅ Server ready at http://127.0.0.1:5000")
    
    try:
        app.run(debug=True, threaded=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        # Cleanup on exit
        cleanup_old_files()
        print("🧹 Final cleanup completed")
