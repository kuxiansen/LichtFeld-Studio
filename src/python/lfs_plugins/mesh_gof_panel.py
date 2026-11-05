# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Gaussian opacity field mesh controls; the work runs on the native collision job."""
import lichtfeld as lf
from . import rml_widgets as w
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .ui import invalidate_panel, RuntimeState, PanelStateBinding
from .panels import panel_class
from .types import Panel

__lfs_panel_classes__ = ["MeshGofPanel"]
__lfs_panel_ids__ = ["lfs.mesh_gof"]

SPECS = {
    "point_voxel": ScrubFieldSpec(.0001, 1, .001, "%.4g"),
    "view_stride": ScrubFieldSpec(1, 64, 1, "%.0f"),
    "min_component_faces": ScrubFieldSpec(0, 10000, 10, "%.0f"),
}
LABELS = {"point_voxel": "mesh_gof.point_voxel", "view_stride": "mesh.view_stride",
          "min_component_faces": "mesh_gof.min_component_faces"}
TOOLTIPS = {"point_voxel": "tooltip.mesh_gof.point_voxel",
            "view_stride": "tooltip.mesh.view_stride",
            "min_component_faces": "tooltip.mesh_gof.min_component_faces"}


@panel_class("mesh_gof")
class MeshGofPanel(Panel):
    def __init__(self):
        self.handle = None
        self.values = dict(point_voxel=.02, view_stride=2, min_component_faces=100, error="")
        self.state = {}
        self.scrubs = ScrubFieldController(
            SPECS, self.values.__getitem__, self.set_value,
            on_error=lambda error: self.set_value("error", lf.ui.tr("preferences.viewport_defaults_invalid")),
        )

    def set_value(self, prop, value):
        self.values[prop] = value
        invalidate_panel(self.handle, "*")

    def generate(self, *_):
        path = lf.ui.save_ply_file_dialog(self.state.get("source_name", "") + "_gof")
        if not path:
            return
        self.values["error"] = ""
        try:
            lf.generate_mesh_gof(path, self.values["point_voxel"], int(self.values["view_stride"]),
                                 int(self.values["min_component_faces"]))
        except (RuntimeError, ValueError) as exc:
            key = getattr(exc, "message_key", "")
            details = getattr(exc, "message_details", "")
            self.values["error"] = (lf.ui.tr(key) + (f"\n{details}" if details else "")) if key else lf.ui.tr("selection.operation_failed_generic")
        except Exception as exc:
            self.values["error"] = str(exc)
        invalidate_panel(self.handle, "*")

    def update(self, doc=None):
        state = lf.get_collision_state()
        changed = state != self.state
        self.state = state
        return self.scrubs.sync_all() or changed

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("mesh_gof")
        self.handle = model.get_handle()
        self.update()
        model.bind_func("panel_label", lambda: lf.ui.tr("mesh_gof.title"))
        model.bind_func("gof_source", lambda: self.state.get("source_name", ""))
        model.bind_func("gof_active", lambda: self.state.get("active", False))
        model.bind_func("gof_progress", lambda: self.state.get("progress", 0))
        model.bind_func("gof_error", lambda: self.values["error"] or self.state.get("error", ""))
        model.bind_func("gof_status", lambda: f'{self.state.get("stage") or lf.ui.tr("training_panel.running")} {self.state.get("progress", 0):.0%}')
        model.bind_event("gof_run", self.generate)
        model.bind_event("gof_cancel", lambda *_: lf.cancel_collision())
        model.bind_event("gof_close", lambda *_: lf.ui.set_panel_enabled("lfs.mesh_gof", False))

    def on_mount(self, doc):
        super().on_mount(doc)
        self.update()
        container = doc.get_element_by_id("gof-fields")
        if container is not None:
            for key, spec in SPECS.items():
                row = w.slider(container, key, label=lf.ui.tr(LABELS[key]), min=spec.min_value, max=spec.max_value,
                               step=spec.step, value=self.values[key])
                row.query_selector(".setting-row__label-col").set_attribute("data-lfs-i18n", LABELS[key])
                row.query_selector("input").set_attribute("data-value", key)
                row.set_attribute("data-tooltip", TOOLTIPS[key])
        self.scrubs.mount(doc)
        self.binding = PanelStateBinding(self.handle).watch(
            RuntimeState.scene_generation, RuntimeState.selection_generation,
            RuntimeState.viewport_toolbar_generation,
            refresh=self.update, dirty="*").watch(RuntimeState.language_generation, dirty="*")

    on_update = update

    def on_unmount(self, doc):
        self.binding.close()
        self.scrubs.unmount()
        self.handle = None
        doc.remove_data_model("mesh_gof")
