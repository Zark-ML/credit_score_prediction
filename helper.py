from loguru import logger

logger.add('logger/log.log', format="{time} {file} {level} {message}", level="INFO", rotation="10 MB")
logger.add('logger/error.log', format="{time} {file} {level} {message}", level="ERROR", rotation="10 MB")
