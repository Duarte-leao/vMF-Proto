"""vMF-MP model package exports."""

from .vmf_mp import VMFMP


def __getattr__(name):
    # Geometry/loss tests do not need the separately installed DINO source trees.
    if name in {
        "DINOv2Backbone",
        "DINOv2BackboneExpanded",
        "DINOv3Backbone",
        "DINOv3BackboneExpanded",
    }:
        from . import backbone

        return getattr(backbone, name)
    if name in {
        "AttnThenPCAForegroundExtractor",
        "AttentionForegroundExtractor",
        "BackboneAttentionForegroundGater",
        "PCA",
    }:
        from . import fg_gaters

        return getattr(fg_gaters, name)
    raise AttributeError(name)


__all__ = [
    "AttnThenPCAForegroundExtractor",
    "AttentionForegroundExtractor",
    "BackboneAttentionForegroundGater",
    "DINOv2Backbone",
    "DINOv2BackboneExpanded",
    "DINOv3Backbone",
    "DINOv3BackboneExpanded",
    "PCA",
    "VMFMP",
]
