from loguru import logger
from pathlib import Path

logger.add('logger/log.log', format="{time} {level} {message}", level="INFO", rotation="10 MB")
logger.add('logger/error.log', format="{time} {level} {message}", level="ERROR", rotation="10 MB")
