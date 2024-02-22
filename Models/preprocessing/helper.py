from loguru import logger
from pathlib import Path

logger.add(Path(__file__).stem + '_log.log', format="{time} {level} {message}", level="INFO", rotation="10 MB")
logger.add(Path(__file__).stem + '_error.log', format="{time} {level} {message}", level="ERROR", rotation="10 MB")
