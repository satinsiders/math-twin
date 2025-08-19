import logging
from twin_generator import cli
from twin_generator.pipeline import PipelineState

def test_verbose_emits_logs(monkeypatch, capsys):
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    for h in old_handlers:
        root.removeHandler(h)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    def fake_generate_twin(*args, **kwargs):
        logging.getLogger().debug("debug message")
        return PipelineState()

    monkeypatch.setattr(cli, "generate_twin", fake_generate_twin)
    try:
        cli.main(["--demo", "--verbose"])
        err = capsys.readouterr().err
        assert "debug message" in err
    finally:
        for h in old_handlers:
            root.addHandler(h)
        root.setLevel(old_level)
