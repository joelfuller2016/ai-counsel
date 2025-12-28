"""Unit tests for config watcher module."""
import os
import tempfile
import threading
import time

import pytest
import yaml

from models.config_watcher import (
    ConfigWatcher,
    ConfigChangeEvent,
    init_config_watcher,
    get_config_watcher,
    shutdown_config_watcher,
)
from models.config import load_config


def create_test_config(tmp_path, **overrides):
    """Create a minimal test config file."""
    base_config = {
        "version": "1.0",
        "adapters": {
            "claude": {
                "type": "cli",
                "command": "claude",
                "args": ["--model", "{model}"],
                "timeout": 60,
            }
        },
        "defaults": {
            "mode": "quick",
            "rounds": 2,
            "max_rounds": 5,
            "timeout_per_round": 120,
        },
        "storage": {
            "transcripts_dir": "transcripts",
            "format": "markdown",
            "auto_export": True,
        },
        "deliberation": {
            "convergence_detection": {
                "enabled": True,
                "semantic_similarity_threshold": 0.85,
                "divergence_threshold": 0.40,
                "min_rounds_before_check": 1,
                "consecutive_stable_rounds": 2,
                "stance_stability_threshold": 0.80,
                "response_length_drop_threshold": 0.40,
            },
            "early_stopping": {
                "enabled": True,
                "threshold": 0.66,
                "respect_min_rounds": True,
            },
            "convergence_threshold": 0.8,
            "enable_convergence_detection": True,
        },
    }

    # Apply overrides
    for key, value in overrides.items():
        base_config[key] = value

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(base_config, f)

    return str(config_path)


class TestConfigWatcher:
    """Tests for ConfigWatcher class."""

    def test_init_loads_config(self, tmp_path):
        """Test ConfigWatcher loads initial config on init."""
        config_path = create_test_config(tmp_path)

        watcher = ConfigWatcher(config_path)

        config = watcher.get_config()
        assert config is not None
        assert config.version == "1.0"

    def test_init_invalid_path_raises(self, tmp_path):
        """Test ConfigWatcher raises error for invalid path."""
        with pytest.raises(FileNotFoundError):
            ConfigWatcher(str(tmp_path / "nonexistent.yaml"))

    def test_get_config_without_load_raises(self, tmp_path):
        """Test get_config raises error if config not loaded."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        # Force config to None (simulating failure)
        watcher._config = None

        with pytest.raises(RuntimeError):
            watcher.get_config()

    def test_add_listener(self, tmp_path):
        """Test adding a change listener."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        events = []
        watcher.add_listener(lambda e: events.append(e))

        assert len(watcher._listeners) == 1

    def test_remove_listener(self, tmp_path):
        """Test removing a change listener."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        callback = lambda e: None
        watcher.add_listener(callback)
        assert len(watcher._listeners) == 1

        watcher.remove_listener(callback)
        assert len(watcher._listeners) == 0

    def test_reload_detects_changes(self, tmp_path):
        """Test reload detects configuration changes."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path, debounce_seconds=0)

        # Modify config file
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        config_data["defaults"]["rounds"] = 5
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Reload
        events = []
        watcher.add_listener(lambda e: events.append(e))
        result = watcher.reload()

        assert result is True
        assert len(events) == 1
        assert "defaults" in events[0].changed_sections

    def test_reload_no_changes(self, tmp_path):
        """Test reload with no changes detected."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path, debounce_seconds=0)

        events = []
        watcher.add_listener(lambda e: events.append(e))

        # Reload without changing the file
        result = watcher.reload()

        assert result is True
        assert len(events) == 0  # No change event fired

    def test_reload_invalid_config_fails_gracefully(self, tmp_path):
        """Test reload fails gracefully with invalid config."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path, debounce_seconds=0)

        original_config = watcher.get_config()

        # Write invalid config
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [broken")

        # Reload should fail but keep old config
        result = watcher.reload()

        assert result is False
        assert watcher.get_config() == original_config

    def test_reload_debounce(self, tmp_path):
        """Test reload respects debounce timing."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path, debounce_seconds=1.0)

        # First reload succeeds
        watcher.reload()

        # Immediate second reload should be debounced
        result = watcher.reload()

        assert result is False

    def test_detect_changed_sections_adapters(self, tmp_path):
        """Test change detection for adapters section."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        old_config = watcher.get_config()

        # Modify adapters
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        config_data["adapters"]["claude"]["timeout"] = 120
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        new_config = load_config(config_path)

        changed = watcher._detect_changed_sections(old_config, new_config)

        assert "adapters" in changed

    def test_detect_changed_sections_defaults(self, tmp_path):
        """Test change detection for defaults section."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        old_config = watcher.get_config()

        # Modify defaults
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        config_data["defaults"]["rounds"] = 4
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        new_config = load_config(config_path)

        changed = watcher._detect_changed_sections(old_config, new_config)

        assert "defaults" in changed

    def test_is_running_false_initially(self, tmp_path):
        """Test is_running returns False before start."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        assert watcher.is_running() is False

    def test_context_manager(self, tmp_path):
        """Test context manager interface."""
        config_path = create_test_config(tmp_path)

        with ConfigWatcher(config_path) as watcher:
            # Watcher should be running inside context
            # Note: may not actually start if watchdog not installed
            config = watcher.get_config()
            assert config is not None

        # Watcher should be stopped after context
        assert watcher.is_running() is False

    def test_stop_when_not_running(self, tmp_path):
        """Test stop() is safe when watcher not running."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        # Should not raise
        watcher.stop()
        assert watcher.is_running() is False


class TestConfigChangeEvent:
    """Tests for ConfigChangeEvent class."""

    def test_event_fields(self, tmp_path):
        """Test ConfigChangeEvent has expected fields."""
        config_path = create_test_config(tmp_path)
        config = load_config(config_path)

        event = ConfigChangeEvent(
            old_config=config,
            new_config=config,
            changed_sections=["defaults", "storage"],
        )

        assert event.old_config is config
        assert event.new_config is config
        assert event.changed_sections == ["defaults", "storage"]
        assert event.timestamp > 0

    def test_event_timestamp_is_current(self, tmp_path):
        """Test ConfigChangeEvent timestamp is approximately current."""
        config_path = create_test_config(tmp_path)
        config = load_config(config_path)

        before = time.time()
        event = ConfigChangeEvent(
            old_config=config,
            new_config=config,
            changed_sections=[],
        )
        after = time.time()

        assert before <= event.timestamp <= after


class TestGlobalWatcher:
    """Tests for global watcher functions."""

    def test_get_config_watcher_none_initially(self):
        """Test get_config_watcher returns None initially."""
        shutdown_config_watcher()  # Reset state

        assert get_config_watcher() is None

    def test_init_config_watcher(self, tmp_path):
        """Test init_config_watcher creates global instance."""
        shutdown_config_watcher()  # Reset state

        config_path = create_test_config(tmp_path)
        watcher = init_config_watcher(config_path)

        assert watcher is not None
        assert get_config_watcher() is watcher

        shutdown_config_watcher()

    def test_shutdown_config_watcher(self, tmp_path):
        """Test shutdown_config_watcher cleans up global instance."""
        shutdown_config_watcher()  # Reset state

        config_path = create_test_config(tmp_path)
        init_config_watcher(config_path)

        shutdown_config_watcher()

        assert get_config_watcher() is None


class TestConfigWatcherThreadSafety:
    """Tests for thread safety of ConfigWatcher."""

    def test_concurrent_get_config(self, tmp_path):
        """Test concurrent get_config calls are thread-safe."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        results = []
        errors = []

        def get_config_task():
            try:
                for _ in range(100):
                    config = watcher.get_config()
                    results.append(config is not None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_config_task) for _ in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(results)

    def test_concurrent_add_remove_listeners(self, tmp_path):
        """Test concurrent listener operations are thread-safe."""
        config_path = create_test_config(tmp_path)
        watcher = ConfigWatcher(config_path)

        errors = []
        callbacks = []

        def listener_task():
            try:
                for _ in range(50):
                    callback = lambda e: None
                    callbacks.append(callback)
                    watcher.add_listener(callback)
                    watcher.remove_listener(callback)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=listener_task) for _ in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
