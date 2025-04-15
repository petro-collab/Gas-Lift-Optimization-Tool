import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Page configuration
st.set_page_config(page_title="Gas-Lift Optimization Tool", layout="wide")

# Title and Description
st.title("Gas-Lift Optimization Tool")
st.markdown("Optimize gas-lift wells based on Redden et al.'s logic. Enter parameters below.")

# Sidebar for Input Parameters
with st.sidebar:
    st.header("Input Parameters")
    reservoir_pressure = st.number_input("Reservoir Pressure (psi)", min_value=0.0, value=3000.0)
    productivity_index = st.number_input("Productivity Index (bbl/day/psi)", min_value=0.0, value=1.0)
    well_depth = st.number_input("Well Depth (ft)", min_value=0.0, value=8000.0)
    tubing_diameter = st.number_input("Tubing Diameter (in)", min_value=0.0, value=2.5)
    surface_pressure = st.number_input("Surface Pressure (psi)", min_value=0.0, value=100.0)
    gas_injection_pressure = st.number_input("Gas Injection Pressure (psi)", min_value=0.0, value=1500.0)
    gas_injection_depth = st.number_input("Gas Injection Depth (ft)", min_value=0.0, value=7000.0)
    gas_specific_gravity = st.number_input("Gas Specific Gravity", min_value=0.0, value=0.7)
    desired_production_rate = st.number_input("Desired Production Rate (bbl/day)", min_value=0.0, value=500.0)

# Define P_wf_range upfront
P_wf_range = np.linspace(0, reservoir_pressure, 100)

# Helper Functions
def calculate_ipr(P_r, J, P_wf_range):
    q_o_max = J * P_r / 1.8
    q_o = [q_o_max * (1 - 0.2 * (P_wf / P_r) - 0.8 * (P_wf / P_r)**2) if P_wf <= P_r else 0 for P_wf in P_wf_range]
    return np.array(q_o)

def calculate_vlp(P_surface, depth, tubing_d, q_l, GLR, gas_sg):
    d = tubing_d / 12
    rho_l = 50
    rho_g = gas_sg * 0.0765
    f_g = GLR / (GLR + 1000)
    rho_m = f_g * rho_g + (1 - f_g) * rho_l
    g = 32.2 / 144
    dP_dz_hydro = rho_m * g
    v = q_l * 5.615 / (86400 * np.pi * (d/2)**2)
    f = 0.02
    dP_dz_friction = f * rho_m * v**2 / (2 * d * 144)
    dP_dz = dP_dz_hydro + dP_dz_friction
    P_wf = P_surface + dP_dz * depth
    return P_wf

def calculate_valve_spacing(P_inj, P_surface, depth, gas_sg, n_valves=3):
    valve_depths = []
    P_tub = P_surface
    delta_depth = depth / (n_valves + 1)
    rho_g = gas_sg * 0.0765 / 144
    for i in range(n_valves):
        z_i = (i + 1) * delta_depth
        P_tub += 0.052 * 50 * delta_depth
        P_inj_z = P_inj + rho_g * z_i
        if P_inj_z > P_tub:
            valve_depths.append(z_i)
        else:
            break
    return valve_depths

def calculate_productivity_index(P_r, P_wf, q):
    return q / (P_r - P_wf) if (P_r - P_wf) != 0 else 0

def calculate_economic_optimum(q_l_range, GLR_range, P_r, J, P_surface, depth, tubing_d, gas_sg, P_wf_range):
    q_o_ipr = calculate_ipr(P_r, J, P_wf_range)
    q_l_opt = []
    for GLR in GLR_range:
        P_wf_values = [calculate_vlp(P_surface, depth, tubing_d, q_l, GLR, gas_sg) for q_l in q_l_range]
        q_o_ipr_interp = interp1d(P_wf_range, q_o_ipr, bounds_error=False, fill_value=0)
        q_l_vlp = [q_o_ipr_interp(P_wf) for P_wf in P_wf_values]
        idx = np.argmin(np.abs(np.array(q_l_vlp) - q_l_range))
        q_l_opt.append(q_l_range[idx])
    return q_l_opt

# Section B: Productivity Index Calculation
st.header("Step B: Productivity Index")
P_wf = calculate_vlp(surface_pressure, well_depth, tubing_diameter, desired_production_rate, 400, gas_specific_gravity)
J = calculate_productivity_index(reservoir_pressure, P_wf, desired_production_rate)
st.write(f"Productivity Index (J) = {J:.2f} bbl/day/psi")

# Section C: Economic Optimum Production Rate
st.header("Step C: Economic Optimum Production Rate")
q_l_range = np.linspace(50, 2000, 50)
GLR_range = np.linspace(100, 1000, 20)
q_l_opt = calculate_economic_optimum(q_l_range, GLR_range, reservoir_pressure, productivity_index,
                                    surface_pressure, well_depth, tubing_diameter, gas_specific_gravity, P_wf_range)
optimal_rate = max(q_l_opt) if q_l_opt else 0
optimal_glr = GLR_range[np.argmax(q_l_opt)] if q_l_opt else 0
st.write(f"Optimal Production Rate = {optimal_rate:.2f} bbl/day")
st.write(f"Optimal GLR = {optimal_glr:.2f} scf/bbl")

# Section D, E, F: Valve Spacing
st.header("Step D, E, F: Valve Spacing")
valve_depths = calculate_valve_spacing(gas_injection_pressure, surface_pressure, gas_injection_depth, gas_specific_gravity)
st.write(f"Valve Depths (ft): {valve_depths}")

# Visualizations
st.header("Visualizations")
q_o_ipr = calculate_ipr(reservoir_pressure, productivity_index, P_wf_range)
q_l_range_plot = np.linspace(50, 2000, 50)
GLR_values = [200, 400, 600, 800]
P_wf_vlp = {GLR: [calculate_vlp(surface_pressure, well_depth, tubing_diameter, q_l, GLR, gas_specific_gravity) for q_l in q_l_range_plot] for GLR in GLR_values}

col1, col2 = st.columns(2)
with col1:
    fig1, ax1 = plt.subplots()
    ax1.plot(q_o_ipr, P_wf_range, label="IPR Curve", color="blue")
    ax1.set_xlabel("Production Rate (bbl/day)")
    ax1.set_ylabel("Bottomhole Pressure (psi)")
    ax1.set_title("IPR Curve")
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    for GLR in GLR_values:
        ax2.plot(q_l_range_plot, P_wf_vlp[GLR], label=f"GLR = {GLR} scf/bbl")
    ax2.plot(q_o_ipr, P_wf_range, label="IPR Curve", color="black", linestyle="--")
    ax2.set_xlabel("Production Rate (bbl/day)")
    ax2.set_ylabel("Bottomhole Pressure (psi)")
    ax2.set_title("VLP Curves")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

st.markdown("### Deploy or Run Locally")
st.markdown("""
- **Local Run**: Install dependencies (`pip install streamlit numpy matplotlib scipy`) and run `streamlit run gas_lift_app.py`.
- **Deploy**: Upload to [Streamlit Community Cloud](https://streamlit.io/cloud) with a GitHub repo and `requirements.txt`.
""")