import smtplib
from email.message import EmailMessage
from decouple import config

GMAIL_USER = config("GMAIL_USER")
GMAIL_PASS = config("GMAIL_PASS")

def send_email(to_email: str, subject: str, body: str):
    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
