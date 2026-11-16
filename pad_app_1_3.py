import streamlit as st
import math
import base64
import pandas as pd

from io import BytesIO
from PIL import Image
from assumptions import get_engineering_assumptions

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Pad Foundation Sizing Tool",
    page_icon="assets/whe_thumbnail.png",
    layout="wide",
)


# -------- CACHED ASSET LOADING (runs once, not on every rerun) --------
@st.cache_resource
def load_css():
    with open("assets/style.css") as f:
        return f.read()


@st.cache_resource
def load_images():
    logo = Image.open("assets/whe_logo_hori.png")
    watermark = Image.open("assets/whe_watermark.png")
    buffer = BytesIO()
    watermark.save(buffer, format="PNG")
    watermark_b64 = base64.b64encode(buffer.getvalue()).decode()
    return logo, watermark_b64


st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)
logo, img_base64 = load_images()

# -------- HEADER --------
col1, col2, col3 = st.columns([2, 2, 0.5])

with col3:
    st.image(logo, width=200)

with col1:
    st.title("Pad Foundation Preliminary Sizing Tool")

# -------- FAINT WATERMARK IN BACKGROUND --------
st.markdown(
    f"""
    <style>
    .stApp::before {{
        background-image: url("data:image/png;base64,{img_base64}");
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------

st.caption("Version v1.4 – Robust solver for high loads")
st.markdown(
    """
    **Purpose**  
    Preliminary sizing of **square pad foundations** under axial load.

    Each pad is checked independently using the same assumptions and rules.

    ⚠️ *Preliminary sizing only.*
    """
)

st.divider()

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
input_col, calc_col, results_col = st.columns([1, 1.2, 1.1])

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "pads" not in st.session_state:
    st.session_state.pads = []

# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

GAMMA_CONC = 24.0  # kN/m³

gamma_G = 1.0
gamma_Q = 1.0

# --------------------------------------------------
# INPUTS (COMMON)
# --------------------------------------------------

with input_col:

    st.header("Global Inputs (apply to all pads)")

    col1, col2 = st.columns(2)

    with col1:
        q_allow = st.number_input(
            "Allowable bearing pressure, qₐ (kN/m²)",
            min_value=25.0,
            value=150.0,
            step=10.0,
        )

        rounding = st.selectbox(
            "Round foundation size up to:",
            options=[0.10, 0.20],
            index=1,
        )

    with col2:
        target_utilisation = st.slider(
            "Target bearing utilisation (%)",
            min_value=50,
            max_value=100,
            value=90,
            step=5,
            help=(
                "Preferred bearing pressure as a % of qₐ. If this can't be "
                "reached, the pad is sized to the smallest size within 100% of qₐ."
            ),
        ) / 100.0

        soil_unit_weight = st.number_input(
            "Soil unit weight, γₛ (kN/m³)",
            min_value=12.0,
            max_value=25.0,
            value=18.0,
            step=0.5,
        )

        pad_effective_weight = st.checkbox(
            "Use effective unit weight (γc - γs) for self-weight calculation",
            value=False,
        )
        if pad_effective_weight:
            pad_unit_weight = GAMMA_CONC - soil_unit_weight
        else:
            pad_unit_weight = GAMMA_CONC

    st.subheader("Minimum Foundation Constraints")

    min_width = st.slider("Minimum pad width (m)", 0.5, 6.0, 1.5, 0.10)
    min_depth = st.slider("Minimum pad depth (m)", 0.3, 2.5, 0.75, 0.05)

    include_self_weight = st.checkbox("Include pad self-weight", value=True)

    st.divider()

    # --------------------------------------------------
    # ADD PAD
    # --------------------------------------------------

    st.header("Add Pad Foundation")

    col1, col2 = st.columns(2)

    with col1:
        G = st.number_input("Dead load, G (kN)", value=3000.0, step=25.0)
        Sur_G = st.number_input(
            "Additional surcharge Dead load (kN/m²)", value=0.0, step=0.5
        )

    with col2:
        Q = st.number_input("Live load, Q (kN)", value=150.0, step=25.0)
        Sur_Q = st.number_input(
            "Additional surcharge Live load (kN/m²)", value=0.0, step=0.5
        )

    col_w = st.number_input(
        "Column / pier width, c (m)", min_value=0.1, value=0.45, step=0.05
    )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("➕ Add pad foundation"):
            next_id = max((p["id"] for p in st.session_state.pads), default=0) + 1
            st.session_state.pads.append(
                {
                    "id": next_id,
                    "G": G,
                    "Q": Q,
                    "Sur_G": Sur_G,
                    "Sur_Q": Sur_Q,
                    "c": col_w,
                }
            )
    with b2:
        if st.button("🗑️ Clear all pads"):
            st.session_state.pads = []

    st.divider()


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def round_up(value, inc):
    # round() guards against float noise, e.g. 1.5/0.1 = 15.000000000000002
    return round(math.ceil(round(value / inc, 9)) * inc, 6)


# --------------------------------------------------
# SOLVER
# --------------------------------------------------
# Geometry rule (unreinforced, 45° load spread):
#   t = max((B - c) / 2, min_depth)   where c = column width
#
# Bearing pressure as a function of B:
#   q(B) = N_base / B²  +  surcharge  +  γG·γpad·t(B)
#
# Because t grows with B, the self-weight term GROWS with B, so q(B) falls,
# hits a minimum, then rises again.
#
# Approach: find the minimum of q(B) analytically, check feasibility, then
# bisect on the falling branch for the smallest B that meets a pressure goal.
#   1. Try the target utilisation (a preference).
#   2. If unreachable, fall back to 100% of qa (the hard limit).
#   3. If even that is unreachable, the pad is infeasible.
# --------------------------------------------------

def solve_pad(
    G,
    Q,
    Sur_G,
    Sur_Q,
    q_allow,
    target_utilisation,
    min_width,
    min_depth,
    col_width,
    include_self_weight,
    pad_unit_weight,
):
    N_base = gamma_G * G + gamma_Q * Q
    q_target = target_utilisation * q_allow
    q_sur = gamma_G * Sur_G + gamma_Q * Sur_Q

    def pressure(B):
        t = max(0.5 * (B - col_width), min_depth)
        q_sw = gamma_G * pad_unit_weight * t if include_self_weight else 0.0
        return N_base / B**2 + q_sur + q_sw

    # Width at which pressure is lowest (end of falling branch)
    if include_self_weight and pad_unit_weight > 0:
        B_star = (4.0 * N_base / (gamma_G * pad_unit_weight)) ** (1.0 / 3.0)
    else:
        B_star = 1e4  # no self-weight: pressure only ever falls

    def smallest_B(goal):
        """Smallest width with pressure <= goal.
        Returns (B or None, lowest achievable pressure, width at that minimum)."""
        B_lo = max(math.sqrt(N_base / goal), min_width, col_width)
        B_hi = max(B_lo, 2.0 * min_depth + col_width, B_star)
        q_best = pressure(B_hi)

        if q_best > goal:
            return None, q_best, B_hi
        if pressure(B_lo) <= goal:
            return B_lo, q_best, B_hi

        lo, hi = B_lo, B_hi
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if pressure(mid) > goal:
                lo = mid
            else:
                hi = mid
        return hi, q_best, B_hi

    # 1. Try the target utilisation (preference)
    B, q_floor, B_at_floor = smallest_B(q_target)
    target_met = B is not None

    # 2. Target unreachable -> fall back to 100% of qa (hard limit)
    if not target_met:
        B, q_floor, B_at_floor = smallest_B(q_allow)

        # 3. Even the best pad exceeds qa -> infeasible
        if B is None:
            return {"infeasible": True, "q_min": q_floor, "B_at_min": B_at_floor}

    B = math.ceil(B * 1000) / 1000  # snap up to 1 mm

    return {
        "B_opt": B,
        "t": max(0.5 * (B - col_width), min_depth),
        "N_ck_initial": N_base,
        "q_target": q_target,
        "q_min": q_floor,
        "target_met": target_met,
    }


# --------------------------------------------------
# CALCULATIONS
# --------------------------------------------------

with calc_col:
    st.header("Pad Foundation Designs")

    for pad in st.session_state.pads:

        st.subheader(f"Pad {pad['id']}")
        pad.pop("results", None)
        c = pad.get("c", 0.45)

        result = solve_pad(
            G=pad["G"],
            Q=pad["Q"],
            Sur_G=pad["Sur_G"],
            Sur_Q=pad["Sur_Q"],
            q_allow=q_allow,
            target_utilisation=target_utilisation,
            min_width=min_width,
            min_depth=min_depth,
            col_width=c,
            include_self_weight=include_self_weight,
            pad_unit_weight=pad_unit_weight,
        )

        if result.get("infeasible"):
            q_min = result["q_min"]
            st.error(
                "No pad size works for this load case. Even the most efficient "
                "pad exceeds the allowable bearing pressure."
            )
            st.markdown(
                f"The lowest pressure any pad can reach is "
                f"**{q_min:.1f} kN/m²** ({q_min / q_allow * 100:.1f}% of qₐ), "
                f"at around B = {result['B_at_min']:.1f} m. Beyond that, extra "
                "width adds pad weight faster than it spreads the load, since "
                "thickness grows with width (t = (B − c)/2).  \n"
                "**Options:** reduce the load or surcharge, increase qₐ, or "
                "tick the effective unit weight option if your qₐ is a net "
                "pressure."
            )
            continue

        B_opt = result["B_opt"]
        N_ck_initial = result["N_ck_initial"]
        q_target = result["q_target"]

        # ---- round UP (conservative) ----
        B_final = round_up(B_opt, rounding)
        t_round = max(0.5 * (B_final - c), min_depth)

        W_pad = B_final**2 * t_round * pad_unit_weight if include_self_weight else 0.0

        N_ck_sur_G = pad["Sur_G"] * B_final**2
        N_ck_sur_Q = pad["Sur_Q"] * B_final**2
        N_ck_surcharge = N_ck_sur_G + N_ck_sur_Q

        N_ck_final = (
            gamma_G * (pad["G"] + W_pad + N_ck_sur_G)
            + gamma_Q * (pad["Q"] + N_ck_sur_Q)
        )

        q_ed = N_ck_final / B_final**2
        utilisation = q_ed / q_allow
        ok = q_ed <= q_allow

        A0 = N_ck_initial / q_target
        B0 = math.sqrt(A0)

        volume = B_final**2 * t_round

        check_line = (
            "✔ Bearing capacity OK"
            if ok
            else "✘ Bearing pressure exceeds qₐ after rounding up – see note above"
        )
        sign = "≤" if ok else ">"

        target_note = (
            ""
            if result["target_met"]
            else (
                f"  \n*Target not achievable for this pad (lowest possible "
                f"utilisation ≈ {result['q_min'] / q_allow * 100:.1f}%). "
                f"The pad is sized to the smallest width within qₐ (100%) instead.*"
            )
        )

        pad["results"] = {
            "B_final": B_final,
            "t_round": t_round,
            "q_ed": q_ed,
            "utilisation": utilisation,
            "N_ck": N_ck_final,
            "volume": volume,
            "calc_md": f"""
**1. Applied loads contributing to foundation design**  

Column dead load, G = {pad['G']:.1f} kN  
Column live load, Q = {pad['Q']:.1f} kN  

Surcharge loads act over the full pad area and are calculated once pad size is known.  
Pad self-weight is included in the design and depends on the adopted pad dimensions.

---

**2. Target bearing pressure**  

q_target = {target_utilisation:.2f} × qₐ  
= {q_target:.1f} kN/m²{target_note}

---

**3. Indicative bearing area and width (excluding size-dependent effects)**  

Required bearing area:  
A = Nck,base / q_target  

Where:  
Nck,base = γG·G + γQ·Q  
= {gamma_G:.2f}×{pad['G']:.1f} + {gamma_Q:.2f}×{pad['Q']:.1f}  
= **{N_ck_initial:.1f} kN**

A = {N_ck_initial:.1f} / {q_target:.1f}  
**A = {A0:.2f} m²**

Indicative pad width:  
B = √A = {B0:.2f} m  

---

**Solving for Size-Dependent Effects**

The pad width is refined to include pad self-weight and surcharge loads (solved directly, not by stepping).

This is then rounded **upwards** to the given rounding, for constructability and conservatism.

The following steps use this adopted size, showing the final design and its checks.

---

**4. Adopted pad geometry**  

Adopted pad width:  
**B = {B_final:.2f} m**

Pad depth governed by 45° load spread (unreinforced pad):  
t = max((B − c) / 2, t_min)  
= max(({B_final:.2f} − {c:.2f}) / 2, {min_depth:.2f})  

Adopted pad depth:  
**t = {t_round:.2f} m**

---

**5. Pad self-weight**  

Pad self-weight calculated using either the full concrete unit weight (γc) or the effective unit weight (γc - γs) if selected.

W = B² · t · γc  
= {B_final:.2f}² × {t_round:.2f} × {pad_unit_weight:.1f}  

**W = {W_pad:.1f} kN**

---

**6. Surcharge loads acting on pad area**  

Dead load surcharge:  
Gₛ = {pad['Sur_G']:.2f} × {B_final:.2f}²  
**Gₛ = {N_ck_sur_G:.1f} kN**

Live load surcharge:  
Qₛ = {pad['Sur_Q']:.2f} × {B_final:.2f}²  
**Qₛ = {N_ck_sur_Q:.1f} kN**

Total surcharge load:  
**Gₛ + Qₛ = {N_ck_surcharge:.1f} kN**

---

**7. Design axial load including self-weight and surcharge**  

Nck = γG·(G + W + Gₛ) + γQ·(Q + Qₛ)

= {gamma_G:.2f}×({pad['G']:.1f} + {W_pad:.1f} + {N_ck_sur_G:.1f})  
+ {gamma_Q:.2f}×({pad['Q']:.1f} + {N_ck_sur_Q:.1f})

**Nck = {N_ck_final:.1f} kN**

---

**8. Bearing pressure check**  

q_ed = Nck / B²  
= {N_ck_final:.1f} / {B_final:.2f}²  

**q_ed = {q_ed:.1f} kN/m² {sign} {q_allow:.1f} kN/m²**

{check_line}
""",
        }

        st.metric("Pad size", f"{B_final:.2f} m × {B_final:.2f} m")
        st.metric("Bearing pressure", f"{q_ed:.1f} kN/m²")
        st.metric("Utilisation", f"{utilisation*100:.1f}%")
        st.metric("Concrete volume", f"{volume:.2f} m³")
        st.metric("Pad depth", f"{t_round:.2f} m")

        if not result["target_met"]:
            st.warning(
                f"The {target_utilisation * 100:.0f}% target can't be reached for "
                f"this pad: its own weight grows with its width, so utilisation "
                f"can't go below about {result['q_min'] / q_allow * 100:.1f}%. "
                f"Sized instead to the smallest pad within qₐ "
                f"({utilisation * 100:.1f}% utilisation).  \n"
                "To reach the target, reduce the load, use the effective unit "
                "weight (if qₐ is net), or increase qₐ."
            )

        if ok:
            st.success("Design OK")
        else:
            st.error(
                f"After rounding up to {rounding:.1f} m, bearing pressure is "
                f"{utilisation * 100:.1f}% of qₐ. Past the lowest-pressure "
                "width, a larger pad adds more weight than it saves, so try a "
                "finer rounding increment or reduce the load."
            )

        with st.expander("Show calculation steps"):
            st.markdown(pad["results"]["calc_md"])

        st.divider()

    # --------------------------------------------------
    # ASSUMPTIONS
    # --------------------------------------------------

st.header("Engineering Assumptions & Design Basis")
with st.expander("Show Assumptions"):
    st.markdown(
        get_engineering_assumptions(
            gamma_G=gamma_G,
            gamma_Q=gamma_Q,
            gamma_conc=GAMMA_CONC,
            min_depth=min_depth,
            min_width=min_width,
            rounding=rounding,
        )
    )

    st.caption(
        "Preliminary sizing only. Not a substitute for full EC7 / EC2 design."
    )

# --------------------------------------------------
# RESULTS TABLE
# --------------------------------------------------
with results_col:
    st.header("Summary of Pad Foundations")

    solved = [p for p in st.session_state.pads if "results" in p]

    if len(st.session_state.pads) == 0:
        st.info("No pad foundations added yet.")
    elif not solved:
        st.warning("No pads could be sized with the current inputs.")
    else:
        summary_data = []
        for pad in solved:
            res = pad["results"]
            summary_data.append(
                {
                    "Pad ID": pad["id"],
                    "Width (m)": round(res["B_final"], 2),
                    "Depth (m)": round(res["t_round"], 2),
                    "Utilisation (%)": round(res["utilisation"] * 100, 1),
                    "SLS Load (kN)": round(res["N_ck"], 1),
                    "Volume (m³)": round(res["volume"], 2),
                }
            )

        st.dataframe(pd.DataFrame(summary_data), width="stretch")

        skipped = len(st.session_state.pads) - len(solved)
        if skipped:
            st.caption(f"{skipped} pad(s) have no feasible size and are excluded.")
