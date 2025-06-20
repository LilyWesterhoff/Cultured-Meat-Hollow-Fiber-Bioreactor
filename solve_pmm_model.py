# solve_pmm_model.py
import numpy as np
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FormatStrFormatter
from waterfall_ax import WaterfallChart
from PIL import Image


# Import from the provided PMM.py
from PMM import hollow_fiber_bioreactor, solve_pv, concentration, visual, visual_2D


def run_simulation(hfb, params):
    """Run the simulation with the given parameters and return results"""
    
    # Solve for velocity and pressure profiles
    domain, lum, ecs, port_info = solve_pv(hfb)
    
    # Solve for steady-state concentration profile 
    conc_list = concentration(hfb, domain, lum, ecs, doubling_t=params['doubling_t'])
    conc = conc_list[0]  # Steady-state concentration (3D array)

    # Compute harvest rates from each port
    x, r, th = domain
    dx = x[0, 0, 1] - x[0, 0, 0]  # Axial step size
    dr = r[0, 1, 0] - r[0, 0, 0] # Radial step size
    dth = th[1, 0, 0] - th[0, 0, 0]  # Angular step size
    vr = 3600 * ecs.face_vel[1]  # Radial velocity at faces [m/h]
    C = conc  # Concentration at centers
    
    # Calculate harvest rate grid (10^6 cells/h per grid cell at r=R where outflow occurs)
    harvest_rate = np.where(vr[:, -1, :] > 0,
                            (vr[:, -1, :] * hfb.R * dth * dx * 1e6 * C[:, -1, :]),
                            0)
    
    # Extract port slices from port_info and sum harvest rates for each port
    up, down = port_info['up'], port_info['down']
    top0, top1 = port_info['top'][0], port_info['top'][1]
    bttm = port_info['bttm']
    harvest_A = np.sum(harvest_rate[top0, up]) + np.sum(harvest_rate[top1, up])  # Port A: Upstream Top
    harvest_B = np.sum(harvest_rate[top0, down]) + np.sum(harvest_rate[top1, down])  # Port B: Downstream Top
    harvest_C = np.sum(harvest_rate[bttm, up])  # Port C: Upstream Bottom
    harvest_D = np.sum(harvest_rate[bttm, down])  # Port D: Downstream Bottom
    
    harvest_rates = {
        'upstream_top': harvest_A,
        'downstream_top': harvest_B,
        'upstream_bottom': harvest_C,
        'downstream_bottom': harvest_D
    }

    # Compute ECS Volume Balances
    # Convert to mL/s for better readability
    conversion = 1e6  # m³/s to mL/s
    flow_A = -(np.sum(vr[top0, -1, up]) + np.sum(vr[top1, -1, up])) * hfb.R * dth * dx * conversion / 3600
    flow_B = -(np.sum(vr[top0, -1, down]) + np.sum(vr[top1, -1, down])) * hfb.R * dth * dx * conversion / 3600
    flow_C = -np.sum(vr[bttm, -1, up]) * hfb.R * dth * dx * conversion / 3600
    flow_D = -np.sum(vr[bttm, -1, down]) * hfb.R * dth * dx * conversion / 3600
    dP = lum.pressure - ecs.pressure
    source = hfb.LP * hfb.a_v * np.sum(np.where(dP>0, dP, 0) *(r*dr*dx*dth)) * 1000 * conversion
    sink = hfb.LP * hfb.a_v * np.sum(np.where(dP<0, dP, 0) *(r*dr*dx*dth)) * 1000 * conversion
    ecs_flows = [source, sink, flow_A, flow_B, flow_C, flow_D]

    # Compute Lumen Volume Balances
    lum_sink = -source
    lum_source = -sink
    lum_in = np.sum(lum.face_vel[0][:, :, 0] * r[:, :, 0] * dr * dth) * conversion
    lum_out = -np.sum(lum.face_vel[0][:, :, -1] * r[:, :, -1] * dr * dth) * conversion
    lum_flows = [lum_in, lum_out, lum_source, lum_sink]

    # Updated return to include harvest_rates
    return domain, lum, ecs, conc, harvest_rates, ecs_flows, lum_flows


def main():
    st.set_page_config(page_title="Continuous Harvest from a Hollow Fiber Bioreactor", layout="wide")
    
    st.title("Modelling Continuous Cell Harvest from a Hollow Fiber Bioreactor")
    st.write("""
    This application simulates steady-state continuous harvest of suspended cells from a hollow-fiber bioreactor based on a porous media model.
    Adjust the parameters and boundary conditions to see how they affect the pressure, velocity, and suspended cell concentration profiles.
    """)

    # Create sidebar for inputs
    with st.sidebar:
        st.header("Bioreactor Parameters")
        R = st.number_input("HFB outer radius [mm]", value=16.0, min_value=1.0) * 1e-3
        L = st.number_input("HFB length [cm]", value=24.0, min_value=1.0) * 1e-2
        angle = st.number_input("HFB angle (0=horizontal, negative=tilted down, positive=tilted up) [degrees]", min_value=-90, max_value=90, value=-45) * math.pi/180
        fib_packing = st.number_input("Fiber packing density [%]", min_value=20, max_value=80, value=77) / 100
        
        st.header("Fiber Parameters")
        R_L = st.number_input("Fiber inner radius [μm]", value=100.0, min_value=10.0) * 1e-6
        R_M = R_L + st.number_input("Fiber membrane thickness [μm]", min_value=10.0, value=40.0) * 1e-6
        mem_pore = st.number_input("Membrane pore size [μm]", value=0.5, min_value=0.1) * 1e-6

        st.header("Cell Parameters")
        doubling_t = st.number_input("Cell doubling time [hours]", value=20.0, min_value=1.0, max_value=100.0)
        cell_r = st.number_input("Cell radius [μm]", value=8.0, min_value=1.0) * 1e-6
        monolayer_voidfrac = st.number_input("Cell layer porosity", min_value=0.1, max_value=0.9, value=0.2)
        cell_dens = st.number_input(r"Adherent Cell Density [$10^6$ cells/mL ECS]", value=170.26)

        # Calculations based on provided parameters
        R_S = R_M/(fib_packing**0.5)
        R_C = (R_M**2 + cell_dens*1e12*(R_S**2 - R_M**2)*(4/3*math.pi*cell_r**3)/(1-monolayer_voidfrac))**0.5 #convert to cells/m3
        cell_layer_thickness = (R_C - R_M) * 1e6 # um
        fluid_layer_thickness = 2 * (R_S - R_C) * 1e6 # um
        monolayer_depth = cell_layer_thickness / (2 * cell_r * 1e6)
        cyl_gap = fluid_layer_thickness * 1e-6 / (2 * cell_r)

        if fluid_layer_thickness<0:
            st.error("You have provided a cell density that exceeds the available ECS volume. Reduce your cell density or your fiber packing density.")
            st.stop()
        
        # Fluid properties
        st.header("Fluid Properties")
        rho = st.number_input("Fluid density [kg/m³]", value=1000.0, min_value=800.0, max_value=1200.0)
        mu = st.number_input("Fluid viscosity [mPa·s]", value=1.0, min_value=0.1, max_value=10.0) * 1e-3
        
        # Boundary conditions
        st.header("Boundary Conditions [kPa/m for Neumann, kPa for Dirichlet]")
        
        # Lumen boundary conditions
        lum_in_type = st.selectbox("Lumen inlet", 
                                    ["Neumann", "Dirichlet"], index=0).lower()
        lum_in_value = st.number_input(
            "Lumen inlet value [kPa/m for neumann, kPa for dirichlet]", 
            value=15.44 if lum_in_type == "neumann" else 105.0,
            label_visibility="collapsed"
        )
        lum_in = (lum_in_type, lum_in_value)
        
        lum_out_type = st.selectbox("Lumen outlet", 
                                    ["Neumann", "Dirichlet"], index=1).lower()
        lum_out_value = st.number_input(
            "Lumen outlet value [kPa/m for neumann, kPa for dirichlet]", 
            value=0.0 if lum_out_type == "neumann" else 103.0,
            label_visibility="collapsed"
        )
        lum_out = (lum_out_type, lum_out_value)
        
        # ECS top upstream port
        ecs_ut_type = st.selectbox("ECS top upstream port (Port A)", ["Neumann", "Dirichlet"], index=0).lower()
        ecs_ut_value = st.number_input(
            "ECS upstream top port value [kPa/m for neumann, kPa for dirichlet]", 
            value=0.0, label_visibility="collapsed"
        )
        ecs_upstreamtop = (ecs_ut_type, ecs_ut_value)
        
        # Downstream top
        ecs_dt_type = st.selectbox("ECS top downstream port (Port B)", ["Neumann", "Dirichlet"], index=1).lower()
        ecs_dt_value = st.number_input(
            "ECS downstream top port value [kPa/m for neumann, kPa for dirichlet]", 
            value=101.0, label_visibility="collapsed"
        )
        ecs_downstreamtop = (ecs_dt_type, ecs_dt_value)

         # Upstream bottom
        ecs_ub_type = st.selectbox("ECS bottom upstream port (Port C)", ["Neumann", "Dirichlet"], index=0).lower()
        ecs_ub_value = st.number_input(
            "ECS upstream bottom port value [kPa/m for neumann, kPa for dirichlet]", 
            value=0.0, label_visibility="collapsed"
        )
        ecs_upstreambttm = (ecs_ub_type, ecs_ub_value)
        
        # Downstream bottom
        ecs_db_type = st.selectbox("ECS bottom downstream port (Port D)", ["Neumann", "Dirichlet"], index=1).lower()
        ecs_db_value = st.number_input(
            "ECS downstream bottom port value [kPa/m for neumann, kPa for dirichlet]", 
            value=101.0, label_visibility="collapsed"
        )
        ecs_downstreambttm = (ecs_db_type, ecs_db_value)

        # =============================================================================
        # Footer
        # =============================================================================

        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        **About this App**

        This Streamlit app provides an interactive interface for exploring continuous cell harvest from a hollow fiber bioreactor. It supplements Westerhoff thesis (2025).
        """)
    
    # Collect all parameters
    params = {
        'R_L': R_L, 'R_M': R_M, 'R': R, 'L': L, 'angle': angle,
        'mem_pore': mem_pore, 'cell_r': cell_r, 
        'monolayer_voidfrac': monolayer_voidfrac, 'monolayer_depth': monolayer_depth,
        'cyl_gap': cyl_gap, 'rho': rho, 'mu': mu, 'doubling_t': doubling_t,
        'lum_in': lum_in, 'lum_out': lum_out,
        'ecs_upstreamtop': ecs_upstreamtop, 'ecs_upstreambttm': ecs_upstreambttm,
        'ecs_downstreamtop': ecs_downstreamtop, 'ecs_downstreambttm': ecs_downstreambttm
    }
    
    # Create the hollow_fiber_bioreactor object with parameters and boundary conditions
    hfb = hollow_fiber_bioreactor(
        R_L=params['R_L'], 
        R_M=params['R_M'], 
        R=params['R'], 
        L=params['L'], 
        angle=params['angle'], 
        mem_pore=params['mem_pore'],
        cell_r=params['cell_r'], 
        monolayer_voidfrac=params['monolayer_voidfrac'], 
        monolayer_depth=params['monolayer_depth'],
        cyl_gap=params['cyl_gap'], 
        rho=params['rho'], 
        mu=params['mu'],
        lum_in=params['lum_in'], 
        lum_out=params['lum_out'],
        ecs_upstreamtop=params['ecs_upstreamtop'], 
        ecs_upstreambttm=params['ecs_upstreambttm'],
        ecs_downstreamtop=params['ecs_downstreamtop'], 
        ecs_downstreambttm=params['ecs_downstreambttm']
    )

    # Spatial domain images
    with st.expander("Spatial Domain and Discretization", expanded=False):

        image = Image.open('images/hfb domain plot.png')
        st.image(image, caption="Schematic of an HFB and the three-dimensional computational domain in cylindrical coordinates.")

        image2 = Image.open('images/point stencil.png')
        st.image(image2, caption="Discretization of representative samples of the spatial domain and example stencil of a Point P and its neighboring points.")
    
    # # Run simulation button
    if st.button("Solve with Provided Parameters") or 'soln' not in st.session_state:
        with st.spinner("Running..."):

            st.markdown("---")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Fibers", f"{hfb.N:,}")
                st.metric("Lumen Axial Permeability [m²]", f"{hfb.k_xl:.2e}")
                st.metric("Cell Layer Thickness [um]", f"{(hfb.R_C - hfb.R_M)*1e6:.2f}")
            with col2:
                st.metric("Total Volume [mL]", f"{math.pi*(hfb.R**2)*hfb.L*1e6:.1f}")
                st.metric("ECS Axial Permeability [m²]", f"{hfb.k_xs:.2e}")
                st.metric("Fluid Layer Thickness [um]", f"{(hfb.R_S - hfb.R_C)*1e6:.2f}")
                
            with col3:
                st.metric("Membrane Hydraulic Conductivity [m/(Pa·s)]", f"{hfb.LP:.2e}")
                st.metric("ECS Radial Permeability [m²]", f"{hfb.k_rs:.2e}")
            
            st.markdown("---")

            try:
                # Run the simulation
                st.session_state['soln'] = run_simulation(hfb, params)
                domain, lum, ecs, conc, harvest_rates, ecs_flows, lum_flows = st.session_state['soln']
                x, r, th = domain
                vS_x, vS_r, vS_th = ecs.center_vel
                vL_x = lum.center_vel[0]

                # Add pie chart for harvest rates
                st.subheader("Cell Harvest Rates by Port")
                
                # Prepare data for pie chart
                port_names = ['Upstream Top (A)', 'Upstream Bottom (C)', 
                             'Downstream Top (B)', 'Downstream Bottom (D)']
                harvest_values = [harvest_rates['upstream_top'], harvest_rates['upstream_bottom'],
                                harvest_rates['downstream_top'], harvest_rates['downstream_bottom']]
                
                # Filter out zero or very small values for better visualization
                filtered_data = [(name, value) for name, value in zip(port_names, harvest_values) if value > 1e-6]
                 
                if filtered_data:
                    names, values = zip(*filtered_data)
                    total_harvest = sum(values)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Create pie chart
                        fig, ax = plt.subplots(figsize=(8, 6))
                        colors = ['#4ECDC4', '#45B7D1', '#96CEB4', '#FF6B6B']
                        wedges, texts, autotexts = ax.pie(values, labels=names, autopct='%1.1f%%', 
                                                         colors=colors[:len(values)], startangle=90)
                        ax.set_title(f'Cell Harvest Rate Distribution\nTotal: {total_harvest:.0f} × 10⁶ cells/h')
                        
                        # Make percentage text more readable
                        for autotext in autotexts:
                            autotext.set_color('white')
                            autotext.set_fontweight('bold')
                        
                        st.pyplot(fig)
                    
                    with col2:
                        # Display harvest rates as metrics
                        st.write("**Harvest Rates [10⁶ cells/h]:**")
                        for name, value in zip(names, values):
                            st.metric(name, f"{value:.0f}")
                        st.metric("**Total Harvest**", f"{total_harvest:.0f}")
                else:
                    st.warning("No significant cell harvest detected from any port. Check boundary conditions and flow patterns.")

                # Add waterfall charts
                st.subheader("Volumetric Flow Rate Balance")

                ecs_vol_balance = [sum(ecs_flows[:i]) for i in range(1, len(ecs_flows)+1)]
                lum_vol_balance = [sum(lum_flows[:i]) for i in range(1, len(lum_flows)+1)]
                ecs_flows.append(sum(ecs_flows))
                lum_flows.append(sum(lum_flows))

                fig2, ax2 = plt.subplots(ncols=2, figsize=(8, 6))
                ecs_names = ['Flow from Lumen', 'Flow to Lumen', 'Port A', 'Port B', 'Port C', 'Port D']
                lum_names = ['Lumen Inlet', 'Lumen Outlet', 'Flow from ECS', 'Flow to ECS']
                waterfall = WaterfallChart(ecs_vol_balance, step_names=ecs_names, 
                                           last_step_label='Balance', metric_name='Volumetric Flow Rate [mL/s]')
                waterfall2 = WaterfallChart(lum_vol_balance, step_names=lum_names, 
                                            last_step_label='Balance', metric_name='Volumetric Flow Rate [mL/s]')
                waterfall.plot_waterfall(ax=ax2[0], title="ECS Volume Balance",
                                         bar_labels=[f"{abs(label):.2f}" for label in ecs_flows],
                                         color_kwargs={'c_bar_pos': 'g', 'c_bar_neg': 'r',
                                                       'c_bar_start': 'g', 'c_bar_end': 'gray'})
                waterfall2.plot_waterfall(ax=ax2[1], title="Lumen Volume Balance",
                                          bar_labels=[f"{abs(label):.2f}" for label in lum_flows],
                                          color_kwargs={'c_bar_pos': 'g', 'c_bar_neg': 'r',
                                                        'c_bar_start': 'g', 'c_bar_end': 'gray'})
                
                for ax in ax2:
                    ax.tick_params(axis='x', labelrotation=90)
                    ax.tick_params(axis='both', labelsize=10)
                    for txt in ax.texts:
                        txt.set_fontsize(8)
                        txt.set_color("black")
                        x_pos, y_pos = txt.get_position()
                        y_min, y_max = ax.get_ylim()
                        txt.set_position((x_pos, max(y_pos, 0.02*(y_max-y_min)))) # Keep bar label above y=0 

                st.pyplot(fig2)

                # Create tabs for different visualizations
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["3D Visualization", "Pressure", "ECS Velocity", 
                                                        "Lumen Velocity", "Concentration"])
                
                with tab1:
                    st.subheader("3D Visualization")
                    
                    # Create 3D visualizations using the visual function
                    st.write("ECS Pressure Distribution (3D)")
                    visual(hfb, domain, ecs.pressure, "Pressure [kPa]", lab_format='%.1f')
                    st.pyplot(plt.gcf())
                    plt.close()

                    st.write("ECS Superficial Velocity Magnitude (3D)")
                    visual(hfb, domain, 10**6 * (vS_x**2 + vS_r**2 + vS_th**2)**0.5, "Velocity [μm/s]", lab_format='%.1f')
                    st.pyplot(plt.gcf())
                    plt.close()

                    st.write("Steady-state Suspended Cell Concentration Distribution (3D)")
                    visual(hfb, domain, conc, r"Concentration [$10^6$ cells/mL]", lab_format='%.1e')
                    st.pyplot(plt.gcf())
                    plt.close()

                with tab2:
                    st.subheader("Pressure at Control Volume Centers")
                    col1, col2 = st.columns(2)

                    for theta in range(12):

                        with col1:
                            visual_2D(x[0, 0, :]*100, r[0, :, 0]*1000, lum.pressure[theta, :, :],
                                    rf'Lumen Pressure $\theta$={(theta+1)/6 - 1/12:.2f}$\pi$ [kPa]')
                            st.pyplot(plt.gcf())
                            plt.close()

                        with col2:
                            visual_2D(x[0, 0, :]*100, r[0, :, 0]*1000, ecs.pressure[theta, :, :],
                                    rf'ECS Pressure $\theta$={(theta+1)/6 - 1/12:.2f}$\pi$ [kPa]')
                            st.pyplot(plt.gcf())
                            plt.close()
                
                with tab3:
                    st.subheader("ECS Superficial Velocity at Control Volume Centers")

                    for theta in range(12):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            visual_2D(x[0, 0, :]*100, r[0, :, 0]*1000, vS_x[theta, :, :]*1e6,
                                    rf'Axial Velocity $\theta$={(theta+1)/6 - 1/12:.2f}$\pi$ [μm/s]')
                            st.pyplot(plt.gcf())
                            plt.close()
                            
                        with col2:
                            visual_2D(x[0, 0, :]*100, r[0, :, 0]*1000, vS_r[theta, :, :]*1e6,
                                    rf'Radial Velocity $\theta$={(theta+1)/6 - 1/12:.2f}$\pi$ [μm/s]')
                            st.pyplot(plt.gcf())
                            plt.close()

                        with col3:
                            visual_2D(x[0, 0, :]*100, r[0, :, 0]*1000, vS_th[theta, :, :]*1e6,
                                    rf'Angular Velocity $\theta$={(theta+1)/6 - 1/12:.2f}$\pi$ [μm/s]')
                            st.pyplot(plt.gcf())
                            plt.close()

                with tab4:
                    st.subheader("Lumen Superficial Velocity at Control Volume Centers")
                    for theta in range(12):
                        visual_2D(x[0, 0, :]*100, r[0, :, 0]*1000, vL_x[theta, :, :]*1e6,
                                rf'Axial Velocity $\theta$={(theta+1)/6 - 1/12:.2f}$\pi$ [μm/s]')
                        st.pyplot(plt.gcf())
                        plt.close()
                            
                with tab5:
                    st.subheader("Steady-state Suspended Cell Concentration at Control Volume Centers")

                    for theta in range(12):

                        visual_2D(x[0, 0, :]*100, r[0, :, 0]*1000, conc[theta, :, :], 
                                rf'Cell Concentration $\theta$={(theta+1)/6 - 1/12:.2f}$\pi$  [$10^6$ cells/mL]')
                        st.pyplot(plt.gcf())
                        plt.close()
                    
            except Exception as e:
                st.error(f"An error occurred during simulation: {str(e)}")
                st.exception(e)



if __name__ == "__main__":
    main()