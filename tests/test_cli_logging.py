import logging
from typing import Any
from twin_generator import cli
from twin_generator.pipeline import PipelineState


def test_log_level_is_isolated(monkeypatch: Any, capsys: Any) -> None:
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    for h in old_handlers:
        root.removeHandler(h)

    pkg_logger = logging.getLogger("twin_generator")
    pkg_old_handlers = pkg_logger.handlers[:]
    pkg_old_level = pkg_logger.level
    for h in pkg_old_handlers:
        pkg_logger.removeHandler(h)

    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def fake_generate_twin(*args, **kwargs):
        logging.getLogger().debug("debug message")
        logging.getLogger("twin_generator").debug("pkg debug")
        return PipelineState()

    monkeypatch.setattr(cli, "generate_twin", fake_generate_twin)
    try:
        cli.main(["--demo", "--log-level", "DEBUG"])
        err = capsys.readouterr().err
        assert "pkg debug" in err
        assert "root debug" not in err
    finally:
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)
        for h in pkg_logger.handlers[len(pkg_old_handlers):]:
            pkg_logger.removeHandler(h)
        for h in pkg_old_handlers:
            pkg_logger.addHandler(h)
        pkg_logger.setLevel(pkg_old_level)
