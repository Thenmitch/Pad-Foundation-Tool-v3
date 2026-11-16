# assumptions.py

# --------------------------------------------------
# ENGINEERING ASSUMPTIONS – PAD FOUNDATIONS
# --------------------------------------------------

def get_engineering_assumptions(
    gamma_G: float,
    gamma_Q: float,
    gamma_conc: float,
    min_depth: float,
    min_width: float,
    rounding: float,
):
    return f"""
### Structural assumptions
- Square unreinforced (plain) concrete pad foundation
- Uniform pad thickness
- Centrally applied axial load only
- No moments or horizontal loads considered
- Uniform bearing pressure distribution assumed

### Load assumptions
- Dead (G) and live (Q) loads applied at column base
- Serviceability limit state load combination:
  
  Nck = γG·(G + W + Gₛ) + γQ·(Q + Qₛ)
- Partial factors:
  - γG = {gamma_G:.2f}
  - γQ = {gamma_Q:.2f}
- Pad self-weight (W) treated as a permanent action
- Concrete unit weight γc = {gamma_conc:.1f} kN/m³
  (optionally reduced to γc − γs, the effective unit weight, if selected)
- Surcharges (Gₛ, Qₛ) are entered as pressures (kN/m²) and act over the full pad area

### Ground assumptions
- Allowable bearing pressure qₐ provided by geotechnical design
- qₐ assumed to already include appropriate safety factors
- Soil assumed homogeneous beneath the foundation
- No settlement or differential settlement assessment carried out

### Geometric assumptions
- Minimum pad depth = {min_depth:.2f} m
- Minimum pad width = {min_width:.2f} m
- Pad thickness governed by 45° load dispersion from the column face:

  t = max((B − c) / 2, t_min)

  where B = pad width, c = column / pier width, t_min = minimum pad depth
- Pad width is solved directly so that bearing pressure, including
  self-weight and surcharge, does not exceed the target utilisation
  (adjustable up to 100% of qₐ). If the target cannot be reached because
  pad weight grows with width, the smallest pad within 100% of qₐ is used
  and a warning is shown
- Foundation widths rounded **upwards only** to {rounding:.2f} m increments, then
  re-checked against the allowable bearing pressure

### Design limitations
The following checks are **not included**:
- Plain concrete tension check (flexural tension in the unreinforced pad,
  EC2 Section 12)
- Sliding, uplift, overturning
- Groundwater effects
- EC7 Design Approach combinations (ULS factored loads)
- Settlement checks

The 45° dispersion rule assumes bearing pressures that are modest relative to
the concrete tensile strength. At high bearing pressures or with low-strength
concrete, a steeper dispersion angle (thicker pad) may be required.

⚠️ This tool is for **preliminary sizing only** and must not be used for final design.
"""
