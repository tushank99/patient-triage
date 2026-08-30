import os
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DATABASE_URL: str = ""
    REDIS_URL: str = ""
    OPENAI_API_KEY: str = ""
    
    # Scalability / Facility Config
    HOSPITAL_TIER: Literal["LEVEL_1_TRAUMA", "COMMUNITY_CLINIC"] = "LEVEL_1_TRAUMA"
    CT_SCANNER_STATUS: Literal["ONLINE", "OFFLINE"] = "ONLINE"
    
    # NLP Settings
    PII_SCRUBBER_ENABLED: bool = True
    
    # Vector DB
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data"

    class Config:
        env_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
        case_sensitive = True
        extra = "ignore"
        


settings = Settings()