from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str
    database_url: str
    redis_url: str
    environment: str = "development"

    # This is the updated Pydantic V2 way to link the .env file
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()  # type: ignore
