"""Tests for CLI argument resolution and config forwarding."""

import json

import pytest
import yaml

from vit_trainer import cli
from vit_trainer.config import TrainingConfig


@pytest.fixture
def yaml_config(tmp_path):
    """Write a config that differs from every argparse default."""

    def _write(**overrides):
        config = TrainingConfig(
            batch_size=128,
            epochs=3,
            lr=5e-5,
            num_workers=0,
            train_split=0.9,
            **overrides,
        )
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(config.to_dict()))
        return path

    return _write


class TestConfigResolution:
    def test_flags_only(self):
        config = cli.resolve_train_config(["train", "--batch-size", "32"])
        assert config.batch_size == 32
        assert config.num_workers == 4  # dataclass default

    def test_yaml_alone(self, yaml_config):
        config = cli.resolve_train_config(["train", "--config", str(yaml_config())])
        assert config.batch_size == 128
        assert config.num_workers == 0
        assert config.train_split == 0.9

    def test_explicit_flag_overrides_yaml(self, yaml_config):
        config = cli.resolve_train_config(
            ["train", "--config", str(yaml_config()), "--batch-size", "8"]
        )
        assert config.batch_size == 8
        assert config.epochs == 3  # untouched YAML value survives

    def test_flag_matching_the_argparse_default_still_wins(self, yaml_config):
        """The case a naive `value != parser default` check gets wrong.

        --batch-size 64 is also argparse's default, but the user typed it, so
        it must override the YAML's 128.
        """
        config = cli.resolve_train_config(
            ["train", "--config", str(yaml_config()), "--batch-size", "64"]
        )
        assert config.batch_size == 64

    def test_store_false_flags(self):
        config = cli.resolve_train_config(["train", "--no-amp", "--no-augment"])
        assert config.use_amp is False
        assert config.augment_train is False

        default = cli.resolve_train_config(["train"])
        assert default.use_amp is True
        assert default.augment_train is True

    def test_dataset_switch_rederives_class_count(self):
        config = cli.resolve_train_config(["train", "--dataset", "cifar100"])
        assert config.num_classes == 100

    def test_explicit_class_count_must_match_dataset(self):
        with pytest.raises(ValueError, match="does not match dataset"):
            cli.resolve_train_config(
                ["train", "--dataset", "cifar100", "--num-classes", "10"]
            )

    def test_unimplemented_dataset_rejected(self):
        """ImageFolder was advertised in config validation but never built."""
        with pytest.raises(ValueError, match="dataset must be one of"):
            TrainingConfig(dataset="imagefolder")

        with pytest.raises(SystemExit):
            cli.resolve_train_config(["train", "--dataset", "imagefolder"])


class _FakeLoader:
    def __init__(self, size=8):
        self.dataset = list(range(size))


class _FakeTrainer:
    """Records what the CLI handed it and skips the actual training."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.metadata = kwargs.get("metadata", {})
        self.history = {}
        self.current_epoch = 0
        _FakeTrainer.instances.append(self)

    def fit(self, train_loader, val_loader, **kwargs):
        self.fit_kwargs = kwargs
        self.history = {
            "train_loss": [0.5, 0.3],
            "val_loss": [0.6, 0.4],
            "lr": [1e-4, 5e-5],
            "epoch_time": [10.0, 11.0],
        }
        return self.history

    def evaluate(self, loader):
        return 0.4, 91.5

    def peak_memory_mb(self):
        return None


@pytest.fixture
def captured_train(monkeypatch, tmp_path):
    """Run cmd_train against stubs, returning what each component received."""
    import vit_trainer.data as data_module
    import vit_trainer.models as models_module
    import vit_trainer.training as training_module

    captured = {}
    _FakeTrainer.instances = []

    def fake_loaders(**kwargs):
        captured["loader_kwargs"] = kwargs
        return _FakeLoader(), _FakeLoader(), _FakeLoader()

    def fake_load_model(variant, **kwargs):
        captured["model_variant"] = variant
        captured["model_kwargs"] = kwargs

        import torch

        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(data_module, "get_cifar10_loaders", fake_loaders)
    monkeypatch.setattr(data_module, "get_cifar100_loaders", fake_loaders)
    monkeypatch.setattr(models_module, "load_model", fake_load_model)
    monkeypatch.setattr(training_module, "Trainer", _FakeTrainer)

    def _run(argv):
        argv = argv + ["--model-dir", str(tmp_path)]
        assert cli.main(argv) == 0
        captured["trainer_kwargs"] = _FakeTrainer.instances[-1].kwargs
        captured["model_dir"] = tmp_path
        return captured

    return _run


class TestTrainForwarding:
    def test_every_config_field_reaches_a_component(self, captured_train):
        """A config field that goes nowhere is the bug this guards against."""
        captured = captured_train(
            [
                "train",
                "--batch-size",
                "16",
                "--num-workers",
                "0",
                "--train-split",
                "0.7",
                "--image-size",
                "160",
                "--no-pin-memory",
                "--no-augment",
                "--gradient-clip",
                "2.5",
                "--seed",
                "7",
                "--lr",
                "3e-4",
                "--weight-decay",
                "0.01",
                "--warmup-epochs",
                "1",
                "--epochs",
                "2",
                "--patience",
                "5",
                "--data-dir",
                "/tmp/data",
            ]
        )

        loader = captured["loader_kwargs"]
        assert loader["batch_size"] == 16
        assert loader["num_workers"] == 0
        assert loader["train_split"] == 0.7
        assert loader["image_size"] == 160
        assert loader["pin_memory"] is False
        assert loader["augment_train"] is False
        assert loader["seed"] == 7
        assert loader["data_dir"] == "/tmp/data"

        trainer = captured["trainer_kwargs"]
        assert trainer["lr"] == 3e-4
        assert trainer["weight_decay"] == 0.01
        assert trainer["warmup_epochs"] == 1
        assert trainer["gradient_clip"] == 2.5

        assert _FakeTrainer.instances[-1].fit_kwargs["epochs"] == 2
        assert _FakeTrainer.instances[-1].fit_kwargs["patience"] == 5

    def test_compile_flag_reaches_the_model_factory(self, captured_train):
        captured = captured_train(["train", "--compile", "--model", "vit_b_32"])
        assert captured["model_variant"] == "vit_b_32"
        assert captured["model_kwargs"]["compile_model"] is True

    def test_yaml_num_workers_reaches_the_loader(self, captured_train, yaml_config):
        captured = captured_train(["train", "--config", str(yaml_config())])
        assert captured["loader_kwargs"]["num_workers"] == 0
        assert captured["loader_kwargs"]["batch_size"] == 128

    def test_run_manifest_is_written(self, captured_train):
        captured = captured_train(["train", "--epochs", "2", "--seed", "7"])
        manifest_path = captured["model_dir"] / "best_model_vit_b_16_cifar10.run.json"
        assert manifest_path.exists()

        manifest = json.loads(manifest_path.read_text())
        assert manifest["config"]["seed"] == 7
        assert manifest["results"]["test_accuracy"] == 91.5
        assert manifest["results"]["best_epoch"] == 2
        assert manifest["results"]["epochs_run"] == 2
        assert manifest["split_hash"]
        assert manifest["environment"]["torch"]
        assert "class_names" in manifest["results"]

    def test_split_hash_is_seed_dependent(self, captured_train, tmp_path):
        seed_a = captured_train(["train", "--seed", "1"])["trainer_kwargs"]
        hash_a = seed_a["metadata"]["split_hash"]
        seed_b = captured_train(["train", "--seed", "2"])["trainer_kwargs"]
        assert hash_a != seed_b["metadata"]["split_hash"]

        seed_a_again = captured_train(["train", "--seed", "1"])["trainer_kwargs"]
        assert hash_a == seed_a_again["metadata"]["split_hash"]


class TestCheckpointDefaults:
    def test_inference_args_read_the_checkpoint(self, tmp_path):
        """eval/predict/export shouldn't need --model repeated by hand."""
        import torch

        path = tmp_path / "ckpt.pt"
        torch.save(
            {
                "model_state_dict": {},
                "metadata": {
                    "config": {
                        "model_variant": "vit_b_32",
                        "dataset": "cifar100",
                        "num_classes": 100,
                    }
                },
            },
            path,
        )

        args = cli.get_parser().parse_args(["eval", "--checkpoint", str(path)])
        variant, dataset, num_classes = cli._resolve_inference_args(args)

        assert (variant, dataset, num_classes) == ("vit_b_32", "cifar100", 100)

    def test_explicit_flags_beat_the_checkpoint(self, tmp_path):
        import torch

        path = tmp_path / "ckpt.pt"
        torch.save(
            {
                "model_state_dict": {},
                "metadata": {"config": {"model_variant": "vit_b_32"}},
            },
            path,
        )

        args = cli.get_parser().parse_args(
            ["eval", "--checkpoint", str(path), "--model", "vit_l_16"]
        )
        assert cli._resolve_inference_args(args)[0] == "vit_l_16"

    def test_legacy_bare_state_dict_still_loads(self, tmp_path):
        """0.1.0 wrote a bare state_dict; those files must keep working."""
        import torch

        from vit_trainer.models import load_state_dict

        path = tmp_path / "legacy.pt"
        weights = {"heads.head.weight": torch.zeros(10, 768)}
        torch.save(weights, path)

        loaded = load_state_dict(str(path), device=torch.device("cpu"))
        assert set(loaded) == set(weights)

        args = cli.get_parser().parse_args(["eval", "--checkpoint", str(path)])
        assert cli._resolve_inference_args(args) == ("vit_b_16", "cifar10", 10)
