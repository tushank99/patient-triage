import re
import logging

logger = logging.getLogger(__name__)

def scrub_pii(text: str) -> str:
    """
    Hackathon Note: In production, this uses Microsoft Presidio/spaCy.
    For the prototype, we use structural regex and simulated NER (Named Entity Recognition).
    """
    if not text:
        return text
        
    # 1. Structural Redaction (SSN, Phone Numbers)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)
    text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE REDACTED]', text)
    
    # 2. Simulated Name Redaction (Catching common names while leaving medical terms)
    # We leave "Parkinson" and "Thomas" (slang) alone if in medical context.
    common_names = ["John", "Jane", "Smith", "Doe", "Mr.", "Mrs."]
    for name in common_names:
        # Simple case-insensitive replacement for the demo
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        text = pattern.sub('[NAME REDACTED]', text)
        
    return text
