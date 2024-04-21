import cv2
import platform
import smtplib
import ssl
import winsound
from email.message import EmailMessage
from email.mime.image import MIMEImage
from datetime import datetime

class AccidentNotification:
    def __init__(self):
        # Define email sender and receiver
        self.email_sender = 'Sender Email Address'
        self.email_password = 'Your Password'
        self.email_receiver = 'Receiver Email Address'

    def send_email(self, subject, body, image_data):
        em = EmailMessage()
        em['From'] = self.email_sender
        em['To'] = self.email_receiver
        em['Subject'] = subject
        em.set_content(body)
        # em.add_attachment(image_data)
        # Add the image data as an attachment with the specified filename
        em.add_attachment(image_data)

        # Add SSL (layer of security)
        context = ssl.create_default_context()

        # Log in and send the email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
            smtp.login(self.email_sender, self.email_password)
            smtp.send_message(em)

    def play_beep(self):
        if platform.system() == "Windows":
            winsound.Beep(1000, 500)  # Frequency, Duration
        else:
            print("No beep functionality for this platform")

    # Main method to trigger accident detection and notification
    def notify_accident(self, frame, latitude, longitude):
        subject = "Accident Detected!"
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        google_maps_link = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

        body = f"An accident has been detected at location (latitude: {latitude}, longitude: {longitude}).\n"
        body += f"Date and Time: {current_datetime}\n"
        body += f"Google Maps Location: {google_maps_link}"

        # Convert the frame to JPEG format
        _, frame_jpeg = cv2.imencode('.jpg', frame)

        # Attach the image to the email
        image_data = MIMEImage(frame_jpeg.tobytes(), name="accident_frame.jpg")
        self.send_email(subject, body, image_data)
        print('Email Sent!')
        self.play_beep()
