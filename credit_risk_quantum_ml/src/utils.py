import logging
from pathlib import Path


def setup_logging(log_file: str = "app.log") -> None:
    """Configura logging básico do projeto."""
    log_path = Path(log_file)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


def handle_exception(exc: Exception) -> None:
    """Tratamento simples de exceções."""
    logging.error("Erro encontrado: %s", exc)
