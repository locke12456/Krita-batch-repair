"""Auto Detect Repair Plugin package.

This package contains the Krita repair plugin shell, detector residency
services, detector execution services, prompt extraction services, generation
handoff services, and layer metadata integration services.
"""

from __future__ import annotations

PLUGIN_ID = "krita_repair_plugin"
PLUGIN_NAME = "Auto Detect Repair Plugin"
PLUGIN_VERSION = "0.1.0"
DOCKER_ID = "kritaRepairAutoDetectDocker"
DOCKER_TITLE = "Auto Detect Repair"

# Keep package import safe in local pytest / non-Krita Python runs.  Runtime
# registration happens only when the real Krita module exposes the required APIs.
try:
    from krita import DockWidgetFactory, DockWidgetFactoryBase, Krita
except Exception:  # pragma: no cover - exercised outside Krita
    DockWidgetFactory = None
    DockWidgetFactoryBase = None
    Krita = None


def _dock_right_position():
    """Resolve Krita 5/6 compatible docker right-side position."""
    if DockWidgetFactoryBase is None:
        return None

    direct = getattr(DockWidgetFactoryBase, "DockRight", None)
    if direct is not None:
        return direct

    for container_name in ("DockPosition", "DockWidgetArea"):
        container = getattr(DockWidgetFactoryBase, container_name, None)
        if container is None:
            continue
        for member_name in ("DockRight", "RightDockWidgetArea", "RightDockWidget"):
            value = getattr(container, member_name, None)
            if value is not None:
                return value

    return None


def _register_krita_plugin() -> None:
    """Register extension fallback actions and the real Krita docker factory."""
    if Krita is None:
        return

    try:
        app = Krita.instance()
    except Exception:
        return

    if app is None:
        return

    if hasattr(app, "addExtension"):
        try:
            from .extension import RepairPluginExtension
        except Exception as exc:
            print(f"{PLUGIN_NAME}: failed to import extension: {exc}")
            RepairPluginExtension = None

        if RepairPluginExtension is not None:
            try:
                app.addExtension(RepairPluginExtension(app))
            except Exception as exc:
                print(f"{PLUGIN_NAME}: failed to register extension: {exc}")

    if DockWidgetFactory is None or DockWidgetFactoryBase is None:
        return

    if hasattr(app, "addDockWidgetFactory"):
        try:
            from .docker import RepairDocker
        except Exception as exc:
            print(f"{PLUGIN_NAME}: failed to import docker: {exc}")
            return

        dock_right = _dock_right_position()
        if dock_right is None:
            print(f"{PLUGIN_NAME}: failed to resolve DockRight position")
            return

        try:
            app.addDockWidgetFactory(
                DockWidgetFactory(
                    DOCKER_ID,
                    dock_right,
                    RepairDocker,
                )
            )
        except Exception as exc:
            print(f"{PLUGIN_NAME}: failed to register docker factory: {exc}")


_register_krita_plugin()


__all__ = [
    "PLUGIN_ID",
    "PLUGIN_NAME",
    "PLUGIN_VERSION",
    "DOCKER_ID",
    "DOCKER_TITLE",
]
