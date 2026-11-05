import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ------------------------------------------------------------------
# Plot formatting
# Set USE_TEX=False if LaTeX is unavailable.
# ------------------------------------------------------------------
USE_TEX = True

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "DejaVu Sans"],
    "font.size": 15,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 2.0,
    "xtick.major.pad": 5,
    "ytick.major.pad": 10,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.size": 10,
    "ytick.major.size": 10,
    "xtick.major.width": 1.25,
    "ytick.major.width": 1.25,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "text.usetex": USE_TEX,
})

if USE_TEX:
    mpl.rcParams["text.latex.preamble"] = r"""
        \usepackage{helvet}
        \usepackage{sfmath}
        \renewcommand{\familydefault}{\sfdefault}
        \usepackage{amsmath}
    """

# ------------------------------------------------------------------
# Load and prepare: one row per host
# ------------------------------------------------------------------
df = pd.read_csv("data/03_trexolists_extended.csv")
df.columns = df.columns.str.strip()

numeric_cols = ["st_teff", "st_rad", "st_logg", "st_met"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)

df["hostname_nn"] = (
    df["hostname_nn"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
)
df = df.dropna(subset=["hostname_nn"])

# Flag inconsistent catalog entries before combining repeated hosts.
conflicts = (
    df.groupby("hostname_nn")[numeric_cols]
    .nunique(dropna=True)
    .gt(1)
)
if conflicts.any(axis=1).any():
    print("Check hosts with inconsistent catalog values:")
    print(conflicts.loc[conflicts.any(axis=1)])

stars = (
    df.groupby("hostname_nn", as_index=False)[numeric_cols]
    .median()
)

# Radius and temperature must be finite and positive.
stars = stars.loc[
    stars["st_teff"].gt(0) & stars["st_rad"].gt(0)
].copy()

if stars.empty:
    raise ValueError("No hosts have valid temperatures and radii.")

# ------------------------------------------------------------------
# Approximate classification
# Unknown log(g) values remain in 'Other / unclassified'.
# ------------------------------------------------------------------
T_MK = 3900
T_KG = 5300

is_dwarf = stars["st_logg"].ge(4.0)

stars["class"] = "Other / unclassified"
stars.loc[
    is_dwarf & stars["st_teff"].lt(T_MK),
    "class",
] = "M dwarf"
stars.loc[
    is_dwarf
    & stars["st_teff"].ge(T_MK)
    & stars["st_teff"].lt(T_KG),
    "class",
] = "K dwarf"

styles = {
    "Other / unclassified": {"marker": "o", "size": 45},
    "K dwarf": {"marker": "s", "size": 85},
    "M dwarf": {"marker": "^", "size": 100},
}

# One common metallicity scale for every spectral group.
feh = stars["st_met"].dropna()
cmap = plt.get_cmap("viridis")

if not feh.empty:
    vmin, vmax = feh.min(), feh.max()

    # Avoid a zero-width color range.
    if np.isclose(vmin, vmax):
        vmin -= 0.1
        vmax += 0.1

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# ------------------------------------------------------------------
# Single-panel figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

tmin = min(stars["st_teff"].min() - 200, T_MK - 200)
tmax = max(stars["st_teff"].max() + 200, T_KG + 200)

# Neutral shading leaves color available for metallicity.
ax.axvspan(tmin, T_MK, color="0.3", alpha=0.055, zorder=0)
ax.axvspan(T_MK, T_KG, color="0.3", alpha=0.025, zorder=0)

for boundary in (T_MK, T_KG):
    ax.axvline(
        boundary, color="0.6", linestyle=":", linewidth=1, zorder=1
    )

for group, style in styles.items():
    subset = stars.loc[stars["class"].eq(group)]
    has_feh = subset["st_met"].notna()

    # Hosts without metallicity.
    missing = subset.loc[~has_feh]
    if not missing.empty:
        ax.scatter(
            missing["st_teff"],
            missing["st_rad"],
            marker=style["marker"],
            s=style["size"],
            facecolors="0.75",
            edgecolors="0.25",
            linewidths=0.7,
            zorder=3,
        )

    # Hosts with measured metallicity.
    measured = subset.loc[has_feh]
    if not measured.empty:
        ax.scatter(
            measured["st_teff"],
            measured["st_rad"],
            c=measured["st_met"],
            cmap=cmap,
            norm=norm,
            marker=style["marker"],
            s=style["size"],
            edgecolors="0.2",
            linewidths=0.7,
            zorder=4 if group != "Other / unclassified" else 2,
        )

ax.set(
    xlabel=r"$T_{\rm eff}$ [K]",
    ylabel=r"$R_\star$ [$R_\odot$]",
    xlim=(tmax, tmin),
    yscale="log",
)
ax.grid(alpha=0.15, which="major")
ax.set_axisbelow(True)

# Spectral-region labels; boundaries are approximate.
for label, left, right in [
    ("M", tmin, T_MK),
    ("K", T_MK, T_KG),
]:
    ax.text(
        (left + right) / 2,
        0.97,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        color="0.4",
        fontsize=13,
    )

# Metallicity colorbar.
if not feh.empty:
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.025)
    cbar.set_label(r"[Fe/H] [dex]")

# Marker-shape legend with unique-host counts.
handles = []
for group, style in styles.items():
    count = stars["class"].eq(group).sum()
    if count == 0:
        continue

    handles.append(
        Line2D(
            [], [],
            linestyle="none",
            marker=style["marker"],
            markerfacecolor="0.65",
            markeredgecolor="0.2",
            markersize=8,
            label=f"{group} ($N={count}$)",
        )
    )

if stars["st_met"].isna().any():
    handles.append(
        Line2D(
            [], [],
            linestyle="none",
            marker="o",
            markerfacecolor="0.75",
            markeredgecolor="0.25",
            markersize=8,
            label="Missing [Fe/H]",
        )
    )

ax.legend(
    handles=handles,
    loc="best",
    frameon=True,
    framealpha=0.9,
    fontsize=10,
)

fig.savefig("Figures4Paper/stellar_properties_feh.pdf", bbox_inches="tight")
fig.savefig("Figures4Paper/stellar_properties_feh.png", dpi=300, bbox_inches="tight")
plt.show()