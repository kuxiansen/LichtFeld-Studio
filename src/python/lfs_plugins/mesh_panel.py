# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Mesh extraction controls; the work runs on the native collision job."""
from pathlib import Path

import lichtfeld as lf
from . import rml_widgets as w
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .ui import invalidate_panel, RuntimeState, PanelStateBinding
from .panels import panel_class
from .types import Panel

__lfs_panel_classes__ = ["MeshPanel"]
__lfs_panel_ids__ = ["lfs.mesh"]

SPECS = {
    "voxel_size": ScrubFieldSpec(0, 10, .001, "%.4g"),
    "view_stride": ScrubFieldSpec(1, 64, 1, "%.0f"),
    "full_weight_range": ScrubFieldSpec(0, 1000, .1, "%.3g"),
    "min_views": ScrubFieldSpec(1, 64, 1, "%.0f"),
    "simplify_error": ScrubFieldSpec(0, .05, .001, "%.3g"),
    "trunc_gradient": ScrubFieldSpec(0, 5, .05, "%.2f"),
    "trunc_cap": ScrubFieldSpec(0, 2, .01, "%.2f"),
    "clean_topology": ScrubFieldSpec(0, 1, 1, "%.0f"),
    "clean_non_manifold": ScrubFieldSpec(0, 1, 1, "%.0f"),
    "remove_spikes": ScrubFieldSpec(0, 1, 1, "%.0f"),
    "spurious_edge_factor": ScrubFieldSpec(0, 100, 1, "%.0f"),
    "spurious_component_factor": ScrubFieldSpec(0, 30, .5, "%.1f"),
    "ear_cut_holes": ScrubFieldSpec(0, 64, 1, "%.0f"),
    "fill_hole_perimeter": ScrubFieldSpec(0, 10, .01, "%.3g"),
    "remove_isolated_small_components": ScrubFieldSpec(0, 1, 1, "%.0f"),
    "isolated_component_max_faces": ScrubFieldSpec(0, 1000, 10, "%.0f"),
    "isolated_component_neighbor_radius": ScrubFieldSpec(0, 50, 1, "%.0f"),
    "max_range": ScrubFieldSpec(0, 1000, 1, "%.3g"),
    "texel_size": ScrubFieldSpec(0, .05, .0005, "%.3g"),
    "texture_size": ScrubFieldSpec(512, 8192, 512, "%.0f"),
    "texture_sharpen": ScrubFieldSpec(0, 2, .05, "%.2g"),
    "lod_levels": ScrubFieldSpec(1, 4, 1, "%.0f"),
}
LABELS = {"voxel_size": "main_panel.voxel_size", "view_stride": "mesh.view_stride",
          "full_weight_range": "mesh.full_weight_range", "min_views": "mesh.min_views",
          "simplify_error": "mesh.simplify_error", "trunc_gradient": "mesh.trunc_gradient",
          "trunc_cap": "mesh.trunc_cap", "clean_topology": "mesh.clean_topology",
          "clean_non_manifold": "mesh.clean_non_manifold",
          "remove_spikes": "mesh.remove_spikes",
          "spurious_edge_factor": "mesh.spurious_edge_factor",
          "spurious_component_factor": "mesh.spurious_component_factor",
          "ear_cut_holes": "mesh.ear_cut_holes",
          "fill_hole_perimeter": "mesh.fill_hole_perimeter",
          "remove_isolated_small_components": "mesh.remove_isolated_small_components",
          "isolated_component_max_faces": "mesh.isolated_component_max_faces",
          "isolated_component_neighbor_radius": "mesh.isolated_component_neighbor_radius",
          "max_range": "mesh.max_range", "texel_size": "mesh.texel_size",
          "texture_size": "mesh.texture_size", "texture_sharpen": "mesh.texture_sharpen",
          "lod_levels": "mesh.lod_levels"}
# Which container each field goes in. The voxel size also sets how far a photo may see past a surface when baking,
# so it stays when the surface comes from a file.
CONTAINERS = {"voxel_size": "mesh-shared-fields", "texel_size": "mesh-texture-fields",
              "texture_size": "mesh-texture-fields", "texture_sharpen": "mesh-texture-fields",
              "lod_levels": "mesh-texture-fields"}


@panel_class("mesh")
class MeshPanel(Panel):
    def __init__(self):
        self.handle = None
        self.values = dict(voxel_size=.0, view_stride=2, full_weight_range=1., min_views=2, simplify_error=.0,
                           trunc_gradient=1.25, trunc_cap=0., clean_topology=1, clean_non_manifold=1,
                           remove_spikes=1,
                           spurious_edge_factor=30., spurious_component_factor=10., ear_cut_holes=50,
                           fill_hole_perimeter=.0,
                           remove_isolated_small_components=1, isolated_component_max_faces=150,
                           isolated_component_neighbor_radius=8.,
                           max_range=0., texel_size=0.,
                           texture_size=4096, texture_sharpen=.5, lod_levels=4,
                           use_file=False, file="", error="")
        self.state = {}
        self.close_when_done = False
        self.scrubs = ScrubFieldController(
            SPECS, self.values.__getitem__, self.set_value,
            on_error=lambda error: self.set_value("error", lf.ui.tr("preferences.viewport_defaults_invalid")),
        )

    def set_value(self, prop, value):
        self.values[prop] = value
        invalidate_panel(self.handle, "*")

    def textured(self):
        return self.values["texel_size"] > 0

    def notice(self):
        """Why the settings cannot run as they are, or empty. A missing file shows on its own row."""
        if self.values["use_file"] and not self.textured():
            return lf.ui.tr("mesh.needs_texel")
        return ""

    def output(self):
        return lf.ui.tr("mesh.output_osgb" if self.textured() else "mesh.output_ply")

    def ready(self):
        if self.values["use_file"] and not self.values["file"]:
            return False
        return bool(self.state.get("source_name")) and not self.state.get("active", False) and not self.notice()

    def choose_file(self, *_):
        start = str(Path(self.values["file"]).parent) if self.values["file"] else ""
        path = lf.ui.open_mesh_file_dialog(start)
        if not path:
            return
        if Path(path).suffix.lower() != ".ply":
            self.values["error"] = lf.ui.tr("mesh.ply_only")
        else:
            self.values["error"] = ""
            self.values["file"] = path
        invalidate_panel(self.handle, "*")

    def generate(self, *_):
        if not self.ready():
            return
        name = Path(self.values["file"]).stem + "_baked" if self.values["use_file"] else self.state.get("source_name", "") + "_mesh"
        path = lf.ui.save_ply_file_dialog(name)
        if not path:
            return
        self.values["error"] = ""
        try:
            lf.generate_mesh(path, self.values["voxel_size"], int(self.values["view_stride"]),
                             self.values["full_weight_range"], int(self.values["min_views"]),
                             self.values["simplify_error"], self.values["max_range"], self.values["texel_size"],
                             self.values["file"] if self.values["use_file"] else "",
                             self.values["trunc_gradient"], self.values["trunc_cap"],
                             int(self.values["clean_topology"]), int(self.values["clean_non_manifold"]),
                             int(self.values["remove_spikes"]), int(self.values["ear_cut_holes"]),
                             self.values["spurious_edge_factor"], self.values["spurious_component_factor"],
                             self.values["fill_hole_perimeter"],
                             int(self.values["remove_isolated_small_components"]),
                             int(self.values["isolated_component_max_faces"]),
                             self.values["isolated_component_neighbor_radius"],
                             int(self.values["texture_size"]),
                             self.values["texture_sharpen"],
                             int(self.values["lod_levels"]))
            self.close_when_done = True
        except (RuntimeError, TypeError, ValueError) as exc:
            key = getattr(exc, "message_key", "")
            details = getattr(exc, "message_details", "")
            self.values["error"] = (lf.ui.tr(key) + (f"\n{details}" if details else "")) if key else lf.ui.tr("selection.operation_failed_generic")
        invalidate_panel(self.handle, "*")

    def cancel(self, *_):
        self.close_when_done = False
        lf.cancel_collision()

    def update(self, doc=None):
        state = lf.get_collision_state()
        changed = state != self.state
        self.state = state
        if self.close_when_done and not state.get("active", False):
            self.close_when_done = False
            if not state.get("error"):
                lf.ui.set_panel_enabled("lfs.mesh", False)
        return self.scrubs.sync_all() or changed

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("mesh")
        self.handle = model.get_handle()
        self.update()
        model.bind_func("panel_label", lambda: lf.ui.tr("mesh.title"))
        model.bind_func("mesh_source", lambda: self.state.get("source_name", ""))
        model.bind_func("mesh_active", lambda: self.state.get("active", False))
        model.bind_func("mesh_progress", lambda: self.state.get("progress", 0))
        model.bind_func("mesh_error", lambda: self.values["error"] or self.state.get("error", ""))
        model.bind_func("mesh_status", lambda: f'{self.state.get("stage") or lf.ui.tr("training_panel.running")} {self.state.get("progress", 0):.0%}')
        model.bind_func("mesh_file", lambda: self.values["file"])
        model.bind_func("mesh_file_name", lambda: Path(self.values["file"]).name if self.values["file"] else lf.ui.tr("mesh.no_file"))
        model.bind_func("mesh_output", self.output)
        model.bind_func("mesh_notice", self.notice)
        model.bind_func("mesh_ready", self.ready)
        model.bind("mesh_use_file", lambda: self.values["use_file"], lambda value: self.set_value("use_file", bool(value)))
        model.bind_event("mesh_choose_file", self.choose_file)
        model.bind_event("mesh_generate", self.generate)
        model.bind_event("mesh_cancel", self.cancel)
        model.bind_event("mesh_close", lambda *_: lf.ui.set_panel_enabled("lfs.mesh", False))

    def on_mount(self, doc):
        super().on_mount(doc)
        self.update()
        for key, spec in SPECS.items():
            container = doc.get_element_by_id(CONTAINERS.get(key, "mesh-fields"))
            if container is None:
                continue
            row = w.slider(container, key, label=lf.ui.tr(LABELS[key]), min=spec.min_value, max=spec.max_value,
                           step=spec.step, value=self.values[key])
            row.query_selector(".setting-row__label-col").set_attribute("data-lfs-i18n", LABELS[key])
            row.query_selector("input").set_attribute("data-value", key)
            row.set_attribute("data-tooltip", "tooltip.mesh." + key)
        self.scrubs.mount(doc)
        self.binding = PanelStateBinding(self.handle).watch(
            RuntimeState.scene_generation, RuntimeState.selection_generation,
            RuntimeState.viewport_toolbar_generation,
            refresh=self.update, dirty="*").watch(RuntimeState.language_generation, dirty="*")

    on_update = update

    def on_unmount(self, doc):
        self.close_when_done = False
        self.binding.close()
        self.scrubs.unmount()
        self.handle = None
        doc.remove_data_model("mesh")
