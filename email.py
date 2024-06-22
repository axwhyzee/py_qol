import smtplib
from email.message import Message
from pathlib import Path
import smtplib
import logging

logger = logging.getLogger(__name__)

SMTP_SERVER = 'smtp.mail.yahoo.com'
SMTP_PORT = 587

import smtplib
from email.message import Message
from pathlib import Path
import smtplib
import logging

logger = logging.getLogger(__name__)

SMTP_SERVER = 'smtp.mail.yahoo.com'
SMTP_PORT = 587

class YahooEmailForwarder:
  
    def __init__(self, sender: str, recipient: str):
        self.sender = sender
        self.recipient = recipient
        self.smtp = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)


    def login(self, password: str):
        """Login using App password, not account password"""
        logger.info('Logging in')
        self.smtp.starttls()
        self.smtp.login(self.sender, password)
        logger.info('Logged in successfully')
        

    def forward(self, eml_path: Path):
        logger.info(f'Forwarding email at {eml_path} ...')
        with open(eml_path) as f:
            msg = email.message_from_file(f)
        msg.replace_header('From', self.sender)
        msg.replace_header('To', self.recipient)
        self.smtp.sendmail(self.sender, self.recipient, msg.as_string())
    
    
    def __enter__(self):
        return self


    def __exit__(self, *_):
        self.smtp.close()
