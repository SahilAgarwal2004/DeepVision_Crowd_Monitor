import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import io
import os
import time

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
IMG_HEIGHT = 256
IMG_WIDTH = 256

# SMTP Configuration Class
class SMTPConfig:
    def __init__(self):
        self.smtp_server = ""
        self.smtp_port = 587
        self.sender_email = ""
        self.sender_password = ""
        self.recipient_emails = []
        self.enabled = False

# Email Alert System
class EmailAlertSystem:
    def __init__(self, config: SMTPConfig):
        self.config = config
        self.last_alert_time = {}
        self.alert_cooldown = 300  # 5 minutes

    def send_alert_email(self, crowd_count, location, crowd_limit, density_map_image=None):
        """Send email alert for crowd limit breach"""
        if not self.config.enabled or not self.config.recipient_emails:
            return False, "SMTP not configured or no recipients"

        current_time = time.time()
        last_alert = self.last_alert_time.get(location, 0)
        if current_time - last_alert < self.alert_cooldown:
            return False, f"Alert cooldown active for {location}"

        try:
            msg = MIMEMultipart('related')
            msg['From'] = self.config.sender_email
            msg['To'] = ', '.join(self.config.recipient_emails)
            msg['Subject'] = f"🚨 CROWD ALERT: {location} - {crowd_count} people detected"

            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; border: 2px solid #ff4444; border-radius: 10px; padding: 20px; background-color: #fff8f8;">
                        <h2 style="color: #ff4444; text-align: center; margin-bottom: 30px;">
                            🚨 CROWD LIMIT EXCEEDED
                        </h2>
                        
                        <div style="background-color: #fff; padding: 20px; border-radius: 5px; margin: 15px 0;">
                            <h3 style="color: #ff4444; margin-top: 0;">Alert Details:</h3>
                            <ul style="list-style-type: none; padding: 0;">
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Location:</strong> {location}
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Current Count:</strong> {crowd_count} people
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Crowd Limit:</strong> {crowd_limit} people
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Exceeded by:</strong> {crowd_count - crowd_limit} people ({((crowd_count/crowd_limit-1)*100):.1f}%)
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; border-left: 4px solid #ff4444;">
                                    <strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                </li>
                            </ul>
                        </div>
                        
                        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin: 20px 0;">
                            <h4 style="color: #856404; margin-top: 0;">⚠️ Recommended Actions:</h4>
                            <ul style="color: #856404;">
                                <li>Implement crowd control measures immediately</li>
                                <li>Consider temporarily restricting entry</li>
                                <li>Ensure adequate safety personnel are present</li>
                                <li>Monitor the situation continuously</li>
                            </ul>
                        </div>
                        
                        <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd;">
                            <p style="color: #666; font-size: 12px;">
                                This is an automated alert from the Crowd Monitoring System<br>
                                Powered by CSRNet Deep Learning Model
                            </p>
                        </div>
                    </div>
                </body>
            </html>
            """

            msg.attach(MIMEText(html_body, 'html'))

            if density_map_image is not None:
                img_attachment = MIMEImage(density_map_image)
                img_attachment.add_header('Content-Disposition', 'attachment', filename='density_map.png')
                msg.attach(img_attachment)

            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.send_message(msg)

            self.last_alert_time[location] = current_time
            return True, "Alert sent successfully"

        except Exception as e:
            return False, f"Failed to send alert: {str(e)}"

    def test_connection(self):
        """Test SMTP connection"""
        try:
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                return True, "SMTP connection successful"
        except Exception as e:
            return False, f"SMTP connection failed: {str(e)}"

# CSRNet Model Definition - MUST MATCH TRAINING CODE EXACTLY
class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        
        # Load VGG16 (NOT VGG16-BN) - matches training code
        vgg = models.vgg16(pretrained=load_weights)
        
        # Frontend: First 23 layers (up to conv5_3) - matches training code
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])
        
        # Backend: Dilated convolutions - matches training code
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(),
        )
        
        # Output layer
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        
        # Upsample if needed
        if x.shape[2:] != (IMG_HEIGHT, IMG_WIDTH):
            x = F.interpolate(x, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=False)
        return x

# Load model
@st.cache_resource
def load_model():
    """Load the trained CSRNet model"""
    model = CSRNet(load_weights=False).to(DEVICE)  # Don't load pretrained VGG
    
    # Try to load your trained weights
    model_paths = [
        "csrnet_best.pth",
        "csrnet_final.pth",
        "/content/drive/MyDrive/models/csrnet_best.pth",
        "models/csrnet_best.pth"
    ]
    
    loaded = False
    for path in model_paths:
        if os.path.exists(path):
            try:
                # Load state dict
                state_dict = torch.load(path, map_location=DEVICE)
                
                # Handle different save formats
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
                
                st.success(f"✅ Loaded trained model from: {path}")
                loaded = True
                break
            except Exception as e:
                st.warning(f"⚠️ Could not load {path}: {e}")
    
    if not loaded:
        st.error("❌ No trained model found. Please upload csrnet_best.pth or csrnet_final.pth")
        st.info("💡 Place your trained model file in the same directory as this script")
    
    return model

# Image preprocessing - MUST MATCH TRAINING CODE
def preprocess_image(image):
    """Preprocess image for model inference - SIMPLE NORMALIZATION ONLY"""
    # Resize image
    img_resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to numpy array
    img_array = np.array(img_resized).astype('float32') / 255.0  # Simple normalization
    
    # Convert to tensor [C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    return img_tensor, img_resized

# Crowd counting prediction
def predict_crowd_count(model, image):
    """Predict crowd count from image"""
    img_tensor, img_resized = preprocess_image(image)
    
    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        crowd_count = int(output.sum().item())
        density_map = output.squeeze().cpu().numpy()
    
    return crowd_count, density_map, img_resized

# Visualization functions
def create_density_visualization(original_img, density_map):
    """Create side-by-side visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.imshow(original_img)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    im = ax2.imshow(density_map, cmap='jet', alpha=0.8)
    ax2.set_title("Predicted Density Map", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def create_crowd_gauge(count, limit):
    """Create a gauge chart for crowd count"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = count,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Crowd Count"},
        delta = {'reference': limit},
        gauge = {
            'axis': {'range': [None, limit * 2]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, limit], 'color': "lightgray"},
                {'range': [limit, limit * 2], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': limit
            }
        }
    ))
    
    fig.update_layout(height=400)
    return fig

def save_density_map_image(fig):
    """Save density map as image bytes"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

# Initialize session state
if 'crowd_history' not in st.session_state:
    st.session_state.crowd_history = []

if 'smtp_config' not in st.session_state:
    st.session_state.smtp_config = SMTPConfig()

if 'email_system' not in st.session_state:
    st.session_state.email_system = EmailAlertSystem(st.session_state.smtp_config)

# Streamlit App
def main():
    st.set_page_config(
        page_title="Crowd Monitoring Dashboard",
        page_icon="👥",
        layout="wide"
    )
    
    st.title("🚨 Crowd Monitoring Dashboard")
    st.markdown("Real-time crowd counting and alert system using CSRNet")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # SMTP Configuration
    with st.sidebar.expander("📧 Email Alert Settings", expanded=False):
        st.subheader("SMTP Configuration")
        
        smtp_server = st.text_input("SMTP Server", 
                                  value=st.session_state.smtp_config.smtp_server,
                                  placeholder="smtp.gmail.com")
        
        smtp_port = st.number_input("SMTP Port", 
                                  value=st.session_state.smtp_config.smtp_port,
                                  min_value=1, max_value=65535)
        
        sender_email = st.text_input("Sender Email", 
                                   value=st.session_state.smtp_config.sender_email,
                                   placeholder="your-email@gmail.com")
        
        sender_password = st.text_input("Sender Password", 
                                      type="password",
                                      placeholder="Enter app password")
        
        recipient_input = st.text_area("Recipient Emails (one per line)",
                                     value='\n'.join(st.session_state.smtp_config.recipient_emails),
                                     placeholder="admin@company.com\nsecurity@company.com")
        
        alerts_enabled = st.checkbox("Enable Email Alerts", 
                                   value=st.session_state.smtp_config.enabled)
        
        if st.button("💾 Save SMTP Settings"):
            st.session_state.smtp_config.smtp_server = smtp_server
            st.session_state.smtp_config.smtp_port = smtp_port
            st.session_state.smtp_config.sender_email = sender_email
            if sender_password:
                st.session_state.smtp_config.sender_password = sender_password
            st.session_state.smtp_config.recipient_emails = [email.strip() for email in recipient_input.split('\n') if email.strip()]
            st.session_state.smtp_config.enabled = alerts_enabled
            st.session_state.email_system = EmailAlertSystem(st.session_state.smtp_config)
            st.success("SMTP settings saved!")
        
        if st.button("🔧 Test SMTP Connection"):
            if st.session_state.smtp_config.enabled:
                with st.spinner("Testing..."):
                    success, message = st.session_state.email_system.test_connection()
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    # Crowd limit
    crowd_limit = st.sidebar.slider("Crowd Limit", min_value=10, max_value=200, value=50, step=5)
    location = st.sidebar.text_input("Location Name", value="Monitoring Area")
    
    with st.sidebar.expander("🚨 Alert Settings", expanded=False):
        auto_alerts = st.checkbox("Automatic Email Alerts", value=True)
        alert_threshold = st.slider("Alert Threshold (%)", min_value=0, max_value=50, value=0, step=5)
    
    # Load model
    model = load_model()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📸 Upload Image for Analysis")
        uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            
            with st.spinner("Analyzing crowd density..."):
                crowd_count, density_map, processed_img = predict_crowd_count(model, image)
            
            effective_limit = crowd_limit + (crowd_limit * alert_threshold / 100)
            should_alert = crowd_count > effective_limit
            
            timestamp = datetime.now()
            st.session_state.crowd_history.append({
                'timestamp': timestamp,
                'count': crowd_count,
                'location': location,
                'is_alert': should_alert,
                'limit': crowd_limit,
                'threshold': effective_limit
            })
            
            # Send alert if needed
            if should_alert and auto_alerts and st.session_state.smtp_config.enabled:
                fig = create_density_visualization(processed_img, density_map)
                density_image = save_density_map_image(fig)
                
                with st.spinner("Sending alert..."):
                    alert_success, alert_message = st.session_state.email_system.send_alert_email(
                        crowd_count=crowd_count,
                        location=location,
                        crowd_limit=crowd_limit,
                        density_map_image=density_image
                    )
                
                if alert_success:
                    st.success("📧 Alert email sent!")
                else:
                    st.warning(f"⚠️ {alert_message}")
            
            # Display results
            st.subheader("📊 Analysis Results")
            
            if should_alert:
                st.error(f"🚨 ALERT: Overcrowded! Count: {crowd_count} (Limit: {int(effective_limit)})")
            else:
                st.success(f"✅ Normal. Count: {crowd_count} (Limit: {crowd_limit})")
            
            fig = create_density_visualization(processed_img, density_map)
            st.pyplot(fig)
    
    with col2:
        st.header("📈 Live Dashboard")
        
        if st.session_state.smtp_config.enabled:
            st.success("📧 Email alerts: ENABLED")
        else:
            st.warning("📧 Email alerts: DISABLED")
        
        if uploaded_file is not None:
            gauge_fig = create_crowd_gauge(crowd_count, crowd_limit)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        st.subheader("📊 Statistics")
        if st.session_state.crowd_history:
            df = pd.DataFrame(st.session_state.crowd_history)
            
            col_stats1, col_stats2 = st.columns(2)
            with col_stats1:
                st.metric("Current Count", crowd_count if uploaded_file else 0)
                st.metric("Total Alerts", len(df[df['is_alert']]))
            
            with col_stats2:
                st.metric("Crowd Limit", crowd_limit)
                st.metric("Total Scans", len(df))
            
            if len(df) > 1:
                st.subheader("📈 History")
                fig_history = px.line(df, x='timestamp', y='count')
                fig_history.add_hline(y=crowd_limit, line_dash="dash", line_color="red")
                st.plotly_chart(fig_history, use_container_width=True)
        
        if st.button("🗑️ Clear History"):
            st.session_state.crowd_history = []
            st.rerun()

if __name__ == "__main__":
    main()