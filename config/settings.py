"""Application configuration via environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """PRISM configuration loaded from environment and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Kalshi
    kalshi_api_key_id: str | None = Field(default=None, alias="KALSHI_API_KEY_ID")
    kalshi_private_key_path: Path | None = Field(
        default=None, alias="KALSHI_PRIVATE_KEY_PATH"
    )
    kalshi_base_url: str = Field(
        default="https://api.elections.kalshi.com/trade-api/v2",
        alias="KALSHI_BASE_URL",
    )

    # Database
    prism_db_path: Path = Field(default=Path("data/prism.duckdb"), alias="PRISM_DB_PATH")

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Data defaults
    nfl_seasons: list[int] = Field(default_factory=lambda: list(range(2018, 2024)))
    nba_seasons: list[int] = Field(default_factory=lambda: list(range(2018, 2024)))

    # Mapping
    match_confidence_threshold: float = 0.8
    ambiguous_match_lower_bound: float = 0.6
