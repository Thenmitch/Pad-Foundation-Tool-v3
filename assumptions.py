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
- Square unreinforced concrete pad foundation
- Uniform pad thickness
- Centrally applied axial load only
- No moments or horizontal loads considered
- Uniform bearing pressure distribution assumed

### Load assumptions
- Dead (G) and live (Q) loads applied at column base
- Serviceability limit state load combination:
  
  Nck = γG·G + γQ·Q
- Partial factors:
  - γG = {gamma_G:.2f}
  - γQ = {gamma_Q:.2f}
- Pad self-weight treated as a permanent action
- Concrete unit weight γc = {gamma_conc:.1f} kN/m³

### Ground assumptions
- Allowable bearing pressure qₐ provided by geotechnical design
- qₐ assumed to already include appropriate safety factors
- Soil assumed homogeneous beneath the foundation
- No settlement or differential settlement assessment carried out

### Geometric assumptions
- Minimum pad depth = {min_depth:.2f} m
- Minimum pad width = {min_width:.2f} m
- 45° load dispersion applied:
  
  B = 2t

- Foundation dimensions rounded **upwards only** to {rounding:.2f} m increments

### Design limitations
The following checks are **not included**:
- Punching shear
- One-way shear
- Flexural reinforcement design
- Crack control
- Sliding, uplift, overturning
- Groundwater effects
- EC7 Design Approach combinations
- Settlement checks

⚠️ This tool is for **preliminary sizing only** and must not be used for final design.
"""
