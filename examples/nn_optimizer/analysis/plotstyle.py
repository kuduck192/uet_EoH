"""Shared matplotlib style for the report figures.

Colours follow the reference palette of the data-viz guide: one fixed colour per
optimizer (identity, never rank), in the validated slot order; scatter plots use
at most the first three slots. The palette has eight slots, so colours are fixed
within each setting (R1/R2: element-wise rules; R3/R4: per-tensor rules), and
full NGD keeps the same colour in both. Adam, the yardstick of every score, is a
neutral dashed line in every figure.

Rules carry the backbone that evolved them: SLM (Qwen3.8-27B, runs R1-R4) or LLM
(Claude Haiku 5.5, runs R1_haiku-R3_haiku).
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
OTHER = "#9a9893"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

ENTITY_ORDER = ["R1#1", "adam", "sgd_momentum", "R2#1", "diag_ngd_emp", "full_ngd", "R1_1_polished",
                "R2_1_polished"]
# R3/R4 avoid the slots of R1#1 and R2#1, so that all four runs differ in the search-progress figure
ENTITY_ORDER_R3 = ["shampoo_cosine", "adam", "adamw_cosine", "muon_cosine", "R3#1", "full_ngd", "R4#1",
                   "R3_1_polished"]
LABELS = {
    "R1#1": "SLM-R1 best", "R1#2": "SLM-R1 #2", "R1#3": "SLM-R1 #3", "R2#1": "SLM-R2 best", "R2#2": "SLM-R2 #2",
    "R2#3": "SLM-R2 #3", "R1@402": "SLM-R1 robust (#402)", "R1_1_polished": "SLM-R1 best, polished",
    "R1@402_polished": "SLM-R1 #402, polished", "R2_1_polished": "SLM-R2 best, polished", "adam": "Adam",
    "sgd_momentum": "SGD-momentum", "lion": "Lion", "signum": "Signum", "diag_ngd_emp": "diag-NGD (EF)",
    "full_ngd": "full NGD", "minimal": "SLM-R1 best, ablated", "R3#1": "SLM-R3 best", "R3#2": "SLM-R3 #2",
    "R3#3": "SLM-R3 #3", "R3@300": "SLM-R3 #6 (dual momentum)", "R3_1_polished": "SLM-R3 best, polished",
    "R4#1": "SLM-R4 best", "R4#2": "SLM-R4 #2", "R4#3": "SLM-R4 #3", "R4_1_polished": "SLM-R4 best, polished",
    "adam_t": "Adam", "adamw_cosine": "AdamW + cosine", "sgd_cosine": "SGD + cosine", "lion_cosine": "Lion + cosine",
    "muon_cosine": "Muon + cosine", "shampoo_cosine": "Shampoo + cosine",
}
# R3_haiku#1, and R3_haiku_1_tolerant: the same rule re-evaluated with exceptions in step() scored as divergence
_LLM_RULE = re.compile(r"^(R\d)_haiku(?:#|_)(\d+)(?:_tolerant)?$")


def is_llm(name):
    return bool(_LLM_RULE.match(name))


def _canon(name):
    return "adam" if name in ("adam_ref", "adam_t") else name


def color(name):
    name = _canon(name)
    if name == "adam":
        return INK2
    for order in (ENTITY_ORDER, ENTITY_ORDER_R3):
        if name in order:
            return SERIES[order.index(name)]
    return OTHER


def linestyle(name):
    return "--" if _canon(name) == "adam" else "-"


def label(name):
    m = _LLM_RULE.match(name)
    if m:
        return f"LLM-{m.group(1)} " + ("best" if m.group(2) == "1" else f"#{m.group(2)}")
    return LABELS.get(_canon(name), name)


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 10, "font.size": 9,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "lines.linewidth": 1.6, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markersize": 5, "legend.frameon": False, "legend.fontsize": 8,
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
