import os

import uvicorn


def _strtobool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    host = os.getenv("UVICORN_HOST", "0.0.0.0")
    port = int(os.getenv("UVICORN_PORT", "8000"))
    workers = int(os.getenv("UVICORN_WORKERS", "1"))
    reload_enabled = _strtobool(os.getenv("UVICORN_RELOAD", "true"))

    if reload_enabled and workers > 1:
        reload_enabled = False

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
