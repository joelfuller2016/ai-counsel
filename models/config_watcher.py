"""Live configuration reload with file watching.

Provides hot-reload capability for config.yaml changes without server restart.
Uses watchdog for cross-platform file system monitoring.
"""
import logging
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

from models.config import Config, load_config

logger = logging.getLogger(__name__)


class ConfigChangeEvent:
    """Event representing a configuration change.

    Attributes:
        old_config: Previous configuration
        new_config: New configuration
        changed_sections: List of section names that changed
        timestamp: Unix timestamp of the change
    """
    def __init__(
        self,
        old_config: Config,
        new_config: Config,
        changed_sections: List[str],
    ):
        self.old_config = old_config
        self.new_config = new_config
        self.changed_sections = changed_sections
        self.timestamp = time.time()


class ConfigWatcher:
    """
    Watches config.yaml for changes and triggers hot-reload.

    Uses watchdog library for cross-platform file system monitoring.
    When changes are detected, validates new config before applying.
    Invalid configurations are rejected with logged warnings.

    Thread-safe: Uses RLock for configuration access.

    Usage:
        watcher = ConfigWatcher("config.yaml")
        watcher.add_listener(lambda event: print(f"Config changed: {event.changed_sections}"))
        watcher.start()

        # Get current config (thread-safe)
        config = watcher.get_config()

        # Manual reload
        watcher.reload()

        # Cleanup
        watcher.stop()
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        debounce_seconds: float = 1.0,
    ):
        """
        Initialize config watcher.

        Args:
            config_path: Path to config.yaml file
            debounce_seconds: Minimum time between reload triggers (prevents rapid reloads)
        """
        self.config_path = Path(config_path)
        self.debounce_seconds = debounce_seconds

        # Thread safety
        self._lock = threading.RLock()
        self._config: Optional[Config] = None
        self._listeners: List[Callable[[ConfigChangeEvent], None]] = []

        # Watchdog state
        self._observer = None
        self._running = False
        self._last_reload_time = 0.0

        # Load initial config
        self._load_initial_config()

    def _load_initial_config(self) -> None:
        """Load the initial configuration."""
        try:
            with self._lock:
                self._config = load_config(str(self.config_path))
            logger.info(f"Initial config loaded from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load initial config: {e}")
            raise

    def get_config(self) -> Config:
        """
        Get the current configuration (thread-safe).

        Returns:
            Current Config object

        Raises:
            RuntimeError: If config was never loaded
        """
        with self._lock:
            if self._config is None:
                raise RuntimeError("Configuration not loaded")
            return self._config

    def add_listener(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """
        Add a listener for config change events.

        Listeners are called when config changes are detected and validated.
        Called in the watcher thread, so listeners should be thread-safe.

        Args:
            callback: Function to call with ConfigChangeEvent
        """
        with self._lock:
            self._listeners.append(callback)
        logger.debug(f"Added config change listener, total: {len(self._listeners)}")

    def remove_listener(self, callback: Callable[[ConfigChangeEvent], None]) -> None:
        """
        Remove a previously added listener.

        Args:
            callback: The callback to remove
        """
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)
                logger.debug(f"Removed config change listener, remaining: {len(self._listeners)}")

    def _detect_changed_sections(self, old: Config, new: Config) -> List[str]:
        """
        Detect which configuration sections changed.

        Args:
            old: Previous configuration
            new: New configuration

        Returns:
            List of changed section names
        """
        changed = []

        # Check top-level sections
        if old.adapters != new.adapters:
            changed.append("adapters")

        if old.cli_tools != new.cli_tools:
            changed.append("cli_tools")

        if old.defaults != new.defaults:
            changed.append("defaults")

        if old.model_registry != new.model_registry:
            changed.append("model_registry")

        if old.storage != new.storage:
            changed.append("storage")

        if old.deliberation != new.deliberation:
            changed.append("deliberation")

        if old.decision_graph != new.decision_graph:
            changed.append("decision_graph")

        return changed

    def reload(self) -> bool:
        """
        Manually trigger a configuration reload.

        Validates new configuration before applying. If validation fails,
        the old configuration is retained.

        Returns:
            True if reload succeeded, False if validation failed

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        current_time = time.time()

        # Debounce check
        if current_time - self._last_reload_time < self.debounce_seconds:
            logger.debug("Reload debounced, too soon after last reload")
            return False

        try:
            # Load new config (validates during load)
            new_config = load_config(str(self.config_path))

            with self._lock:
                old_config = self._config

                # Detect changes
                changed_sections = self._detect_changed_sections(old_config, new_config)

                if not changed_sections:
                    logger.debug("Config reloaded but no changes detected")
                    self._last_reload_time = current_time
                    return True

                # Apply new config
                self._config = new_config
                self._last_reload_time = current_time

                # Create change event
                event = ConfigChangeEvent(
                    old_config=old_config,
                    new_config=new_config,
                    changed_sections=changed_sections,
                )

                # Notify listeners
                listeners = list(self._listeners)

            # Call listeners outside lock to prevent deadlocks
            logger.info(f"Config reloaded, changed sections: {changed_sections}")
            for listener in listeners:
                try:
                    listener(event)
                except Exception as e:
                    logger.error(f"Config change listener error: {e}", exc_info=True)

            return True

        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            raise
        except Exception as e:
            logger.warning(
                f"Config reload failed, keeping previous config: {e}",
                exc_info=True
            )
            return False

    def start(self) -> None:
        """
        Start watching for config file changes.

        Uses watchdog for file system monitoring. Changes are debounced
        to prevent rapid reloads during file saves.
        """
        if self._running:
            logger.warning("ConfigWatcher already running")
            return

        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileModifiedEvent

            watcher = self

            class ConfigFileHandler(FileSystemEventHandler):
                """Handler for config file changes."""

                def on_modified(self, event):
                    if event.is_directory:
                        return

                    # Check if it's our config file
                    event_path = Path(event.src_path).resolve()
                    config_path = watcher.config_path.resolve()

                    if event_path == config_path:
                        logger.debug(f"Config file change detected: {event.src_path}")
                        try:
                            watcher.reload()
                        except Exception as e:
                            logger.error(f"Failed to reload config: {e}")

            # Create and start observer
            self._observer = Observer()
            handler = ConfigFileHandler()

            # Watch the directory containing config file
            watch_dir = str(self.config_path.parent.resolve())
            self._observer.schedule(handler, watch_dir, recursive=False)
            self._observer.start()
            self._running = True

            logger.info(f"Config watcher started for {self.config_path}")

        except ImportError:
            logger.warning(
                "watchdog not installed, live config reload disabled. "
                "Install with: pip install watchdog"
            )
        except Exception as e:
            logger.error(f"Failed to start config watcher: {e}", exc_info=True)

    def stop(self) -> None:
        """
        Stop watching for config file changes.

        Safely shuts down the watchdog observer.
        """
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
            self._running = False
            logger.info("Config watcher stopped")

    def is_running(self) -> bool:
        """Check if watcher is currently running."""
        return self._running

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


# Global watcher instance for server-wide access
_global_watcher: Optional[ConfigWatcher] = None


def get_config_watcher() -> Optional[ConfigWatcher]:
    """Get the global config watcher instance."""
    return _global_watcher


def init_config_watcher(config_path: str = "config.yaml") -> ConfigWatcher:
    """
    Initialize the global config watcher.

    Args:
        config_path: Path to config.yaml

    Returns:
        Initialized ConfigWatcher instance
    """
    global _global_watcher
    if _global_watcher is None:
        _global_watcher = ConfigWatcher(config_path)
    return _global_watcher


def shutdown_config_watcher() -> None:
    """Shutdown the global config watcher."""
    global _global_watcher
    if _global_watcher is not None:
        _global_watcher.stop()
        _global_watcher = None
