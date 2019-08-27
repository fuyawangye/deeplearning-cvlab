import logging.config

config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
        # 其他的 formatter
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'simple'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logging.log',
            'level': 'DEBUG',
            'formatter': 'simple'
        },
        # 其他的 handler
    },
    'loggers':{
        'StreamLogger': {
            'handlers': ['console'],
            'level': 'DEBUG',
        },
        'FileLogger': {
            # 既有 console Handler，还有 file Handler
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        },
        # 其他的 Logger
    }
}


def get_logger(logtype, level="INFO", logfile=""):
    if logtype == "stream":
        config["loggers"]["StreamLogger"]["level"] = level
        logging.config.dictConfig(config)
        return logging.getLogger("StreamLogger")
    else:
        config["loggers"]["FileLogger"]["level"] = level
        config["handlers"]["file"]["filename"] = logfile
        logging.config.dictConfig(config)
        return logging.getLogger("FileLogger")
