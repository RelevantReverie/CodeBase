"""Phase 8 - model registry. One interface for all six GCNs.

    from phase8_models import build_model, MODEL_NAMES
    model = build_model("ctr_gcn", num_class=100, graph=Graph())

Every model takes (num_class, graph, **kw) and maps input [N, C, T, V, M] -> logits.
"""
from .st_gcn import build_model as _st_gcn
from .ctr_gcn import build_model as _ctr_gcn
from .ms_g3d import build_model as _ms_g3d
from .info_gcn import build_model as _info_gcn
from .hd_gcn import build_model as _hd_gcn
from .ha_gcn import build_model as _ha_gcn

MODELS = {
    "st_gcn": _st_gcn,
    "ms_g3d": _ms_g3d,
    "ctr_gcn": _ctr_gcn,
    "info_gcn": _info_gcn,
    "hd_gcn": _hd_gcn,
    "ha_gcn": _ha_gcn,
}
MODEL_NAMES = list(MODELS)


def build_model(name, num_class, graph, **kw):
    if name not in MODELS:
        raise KeyError(f"unknown model '{name}'; choose from {MODEL_NAMES}")
    return MODELS[name](num_class, graph, **kw)
