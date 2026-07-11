import logging
from app.settings import get_settings

def configure_logging() -> None:
    level = getattr(logging, get_settings().log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
