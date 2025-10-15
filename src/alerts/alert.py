"""
Email Alert System for Crowd Monitoring
Handles SMTP configuration and alert emails
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
import time

class SMTPConfig:
    """SMTP Configuration"""
    def __init__(self):
        self.smtp_server = ""
        self.smtp_port = 587
        self.sender_email = ""
        self.sender_password = ""
        self.recipient_emails = []
        self.enabled = False

class EmailAlertSystem:
    """Email alert system for crowd limit breaches"""
    
    def __init__(self, config: SMTPConfig):
        self.config = config
        self.last_alert_time = {}  # Track last alert per location
        self.alert_cooldown = 300  # 5 minutes cooldown

    def send_alert_email(self, crowd_count, location, crowd_limit, density_map_image=None):
        """
        Send email alert for crowd limit breach
        
        Args:
            crowd_count: Current crowd count
            location: Location name
            crowd_limit: Maximum allowed crowd
            density_map_image: Optional density map image bytes
            
        Returns:
            (success, message) tuple
        """
        if not self.config.enabled or not self.config.recipient_emails:
            return False, "SMTP not configured or no recipients"

        # Check cooldown
        current_time = time.time()
        last_alert = self.last_alert_time.get(location, 0)
        if current_time - last_alert < self.alert_cooldown:
            return False, f"Alert cooldown active for {location}"

        try:
            # Create message
            msg = MIMEMultipart('related')
            msg['From'] = self.config.sender_email
            msg['To'] = ', '.join(self.config.recipient_emails)
            msg['Subject'] = f"🚨 CROWD ALERT: {location} - {crowd_count} people detected"

            # HTML body
            html_body = f"""
            <html>
                <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                    <div style="max-width: 600px; margin: 0 auto; border: 2px solid #ff4444; 
                                border-radius: 10px; padding: 20px; background-color: #fff8f8;">
                        
                        <h2 style="color: #ff4444; text-align: center; margin-bottom: 30px;">
                            🚨 CROWD LIMIT EXCEEDED
                        </h2>
                        
                        <div style="background-color: #fff; padding: 20px; border-radius: 5px; margin: 15px 0;">
                            <h3 style="color: #ff4444; margin-top: 0;">Alert Details:</h3>
                            <ul style="list-style-type: none; padding: 0;">
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; 
                                          border-left: 4px solid #ff4444;">
                                    <strong>Location:</strong> {location}
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; 
                                          border-left: 4px solid #ff4444;">
                                    <strong>Current Count:</strong> {crowd_count} people
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; 
                                          border-left: 4px solid #ff4444;">
                                    <strong>Crowd Limit:</strong> {crowd_limit} people
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; 
                                          border-left: 4px solid #ff4444;">
                                    <strong>Exceeded by:</strong> {crowd_count - crowd_limit} people 
                                    ({((crowd_count/crowd_limit-1)*100):.1f}%)
                                </li>
                                <li style="margin: 10px 0; padding: 8px; background-color: #f9f9f9; 
                                          border-left: 4px solid #ff4444;">
                                    <strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                                </li>
                            </ul>
                        </div>
                        
                        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; 
                                    padding: 15px; border-radius: 5px; margin: 20px 0;">
                            <h4 style="color: #856404; margin-top: 0;">⚠️ Recommended Actions:</h4>
                            <ul style="color: #856404;">
                                <li>Implement crowd control measures immediately</li>
                                <li>Consider temporarily restricting entry</li>
                                <li>Ensure adequate safety personnel are present</li>
                                <li>Monitor the situation continuously</li>
                            </ul>
                        </div>
                        
                        <div style="text-align: center; margin-top: 30px; padding-top: 20px; 
                                    border-top: 1px solid #ddd;">
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

            # Attach density map if provided
            if density_map_image is not None:
                img_attachment = MIMEImage(density_map_image)
                img_attachment.add_header('Content-Disposition', 'attachment', 
                                        filename='density_map.png')
                msg.attach(img_attachment)

            # Send email
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.sender_email, self.config.sender_password)
                server.send_message(msg)

            # Update last alert time
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


# Example usage
if __name__ == "__main__":
    # Configure SMTP
    config = SMTPConfig()
    config.smtp_server = "smtp.gmail.com"
    config.smtp_port = 587
    config.sender_email = "your-email@gmail.com"
    config.sender_password = "your-app-password"
    config.recipient_emails = ["admin@company.com"]
    config.enabled = True
    
    # Create alert system
    alert_system = EmailAlertSystem(config)
    
    # Test connection
    success, message = alert_system.test_connection()
    print(f"Connection test: {message}")
    
    # Send test alert
    if success:
        success, message = alert_system.send_alert_email(
            crowd_count=75,
            location="Main Entrance",
            crowd_limit=50
        )
        print(f"Alert test: {message}")