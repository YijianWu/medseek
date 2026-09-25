from pathlib import Path

from medseek.config import load_experiment_config, model_config_from_experiment


CONFIG = Path(__file__).parents[1] / "configs" / "medseek.yaml"


def test_public_config_disables_optional_auxiliary_tasks() -> None:
    config = model_config_from_experiment(load_experiment_config(CONFIG))
    assert config.diagnosis_aux_enabled is False
    assert config.diagnosis_prototype_path is None
    assert config.secondary_endpoint_cls_enabled is False
    assert config.secondary_endpoint_margin_enabled is False
