import logging

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s [%(name)-15s] :: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
    },
    "loggers": {
        # This is the logger from your file
        "user_routes": {"handlers": ["default"], "level": "INFO", "propagate": False},
        
        # Add entries for all your other loggers here
        "search_books_api": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "library_info_route": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "query_router": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "chat_retention": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "login": {"handlers": ["default"], "level": "INFO", "propagate": False},

        # Configure uvicorn's loggers to use our handler
        "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["default"], "level": "INFO", "propagate": False},
    },
}