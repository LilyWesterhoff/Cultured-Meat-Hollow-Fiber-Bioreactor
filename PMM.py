# PMM_Final.py
"""
Model of a hollow-fiber bioreactor based on porous media model as described by
Labecki (1994). Module includes the hollow_fiber_bioreactor class definition,
pressure and velocity solver function (solve_pv), a volume balance check 
function (vol_bal_check), suspended cell concentration solver function 
(concentration), and a plotting function (visual). 
"""

# %% Imports
import numpy as np
from scipy import sparse
import math
from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# %% Class and Function Definitions

class hollow_fiber_bioreactor:

    def __init__(self,
                 R_L = 1e-4,
                 R_M = 1.4e-4,
                 R = 0.016,
                 L = 0.24,
                 angle = -0.25*math.pi,
                 mem_pore = 0.5e-6,
                 cell_rho = 1030,
                 cell_r = 8e-6,
                 monolayer_voidfrac = 0.2,
                 monolayer_depth = 0.575,
                 cyl_gap = 1.25,
                 rho = 1000,
                 mu = 1e-3,
                 lum_in = ('neumann', 15.44),
                 lum_out = ('dirichlet', 101),
                 ecs_upstreamtop = ('neumann', 0),
                 ecs_upstreambttm = ('neumann', 0),
                 ecs_downstreamtop = ('neumann', 0),
                 ecs_downstreambttm = ('dirichlet', 101)
                 ):
        """
        Creates hollow fiber bioreactor objects.
        
        Boundary conditions needed for solving for the pressure and velocity
        profiles are set here as well:
            
            Neumann boundary conditions (negative=outflow, positive=inflow) 
            should be given in [kPa/m] and Dirichlet should be given in [kPa]. 
            For ECS radial boundary conditions at r=R, Neumann BC of 0 already 
            built in so only need to define exceptions to this condition. 
            Neumann BC of 0 is built in for ECS axial boundary conditions as 
            well. Angular boundary conditions are symmetry at theta=0 and 
            theta=pi (which also means that ecs port locations are symmetrical 
            about that plane as well (ports on top or bottom).
            
            Default Neumann BC for lumen inlet is calculated from Q_lum based 
            on FiberCell max of 200mL/min for C2018:
                dP/dx = (Q_lum/CSA)*(mu/k_xl)
                Q_lum/CSA = 200*1e-6*(1/60) / (3.14*0.016**2)
                mu/k_xl = 1e-6 / ((1e-4**2/8)*(5500*1e-4**2/0.016**2))

        Parameters
        ----------
        R_L : float, default 1e-4
            Lumen radius [m]. Default from Fibercell.
        R_M : float, default 1.4e-4
            Outer fiber radius [m]. Default from Fibercell.
        R : float, default 0.016
            Bioreactor radius [m]. Default from Fibercell.
        L : float, default 0.24
            Spatial domain axial length [m]. Default from Fibercell.
        angle : float, default 0.25*math.pi
            HFB angle relative to gravity vector [radians]. Default is 
            angled down 45 degrees.
        mem_pore : float, default 0.5e-6
            Membrane pore radius [m]. Default from Fibercell.
        cell_r : float, default 8e-6
            Cell radius [m].
        monolayer_voidfrac : float, default 0.2
            Fixed void fraction (porosity) of cell layer around fibers.
        monolayer_depth : float, default 0.75
            Monolayer depth around each fiber relative to cell diameter.
        cyl_gap : float, default 3
            Total gap between cell covered fibers relative to cell diameter.
        rho : float, default 1000
            Fluid density [kg/m3].
        mu : float, default 1e-3
            Fluid dynamic viscosity [Pa s].
        lum_in : tuple, default ('neumann', 15.44)
            Lumen inlet axial boundary condition.
        lum_out : tuple, default ('dirichlet', 101) 
            Lumen outlet axial boundary condition.
        ecs_upstreamtop : tuple, default ('neumann', 0)
            ECS upstream top port boundary condition, default is closed.
        ecs_upstreambttm : tuple, default ('neumann', 0)
            ECS upstream bottom port boundary condition, default is closed.
        ecs_downstreamtop : tuple, default ('neumann', 0)
            ECS downstream top port boundary condition, default is closed.
        ecs_downstreambttm : tuple, default ('dirichlet', 101)
            ECS downstream bottom port boundary condition, default is open with
            dirichlet boundary condition.

        Returns
        -------
        None.

        """
        self.__dict__.update(locals())

    # Read-only Calculated Properties (could add setter)
    @property
    def N(self):
        """
        Number of fibers (rounded up to nearest whole number) based on volume
        taken up by each individual fiber (including cell monolayer and gap)
        and the hollow fiber bioreactor radius.
        """
        return math.ceil((self.R/self.R_S)**2)
    
    @property
    def fiber_packing_density(self):
        """
        Fiber packing density of hollow fiber bioreactor.
        """
        return self.R_M**2 / self.R_S**2

    @property
    def R_S(self):
        """
        'Krogh cylinder' radius [m] (used to ensure representative elementary
        volume includes multiple fibers).
        """
        return self.R_C + self.cyl_gap*self.cell_r 

    @property
    def R_C(self):
        """
        Outer radius of cell-covered fibers [m].
        """
        return self.R_M + self.monolayer_depth*2*self.cell_r

    @property
    def cell_dens(self):
        """
        Cell density [cells/mL bioreactor volume].
        """
        single_cell_vol = (4/3)*math.pi*(self.cell_r*1e2)**3  # cm^3
        cells_vfrac = ((1-self.monolayer_voidfrac) * (self.R_C**2-self.R_M**2)
                       / (self.R_S**2))
        return cells_vfrac / single_cell_vol

    @property
    def a_v(self):
        """
        Total inner fiber surface area per bioreactor volume [1/m].
        """
        return (2 * self.R_L * self.N) / (self.R**2)

    @property
    def k_rm(self, C=6.54e-4):
        """
        Radial membrane permeability [m2].
        """
        return C*self.mem_pore**2

    @property
    def k_rc(self, kozeny_constant=5):
        """
        Cell layer radial permeability [m2].
        """
        # Surface area to volume ratio of a sphere [1/m]
        area_vol = 3/self.cell_r
        k = (1/kozeny_constant)*((self.monolayer_voidfrac**3)
                                 /((area_vol*(1-self.monolayer_voidfrac))**2))
        return k

    @property
    def k_rmc(self):
        """
        Harmonic average of cell layer and membrane radial permeabilities [m2].
        """
        return math.log(self.R_C/self.R_L)/(
            (math.log(self.R_M/self.R_L)/self.k_rm)
            + math.log(self.R_C/self.R_M)/self.k_rc)

    @property
    def LP(self):
        """
        Membrane hydraulic conductivity [m/(Pa s)]
        """
        return self.k_rmc / (self.mu*self.R_L*math.log(self.R_C/self.R_L))

    @property
    def k_xl(self):
        """
        Lumen axial permeability [m2].
        """
        porosity_L = (self.N * self.R_L**2) / (self.R**2)
        return (self.R_L**2 / 8) * porosity_L

    @property
    def capillary_d(self):
        """
        Idealized cell layer porous medium pore diameter (from the derivation 
        of the Kozeny-Carman equation) [m].
        """
        return 2*((2*self.monolayer_voidfrac)/(
            (3/(self.cell_r))*(1-self.monolayer_voidfrac)))
    
    @property
    def cyl_packing(self):
        return self.R_C**2/self.R_S**2

    @property
    def k_xs(self):
        """
        Shell-side (cell-free fluid layer) axial permeability [m2]. 
        """
        cyl_packing = self.cyl_packing
        k_xcyl = ((self.R_C**2)/(4*cyl_packing)) * (
            -math.log(cyl_packing)-1.5+2*cyl_packing-0.5*cyl_packing**2)
        return max(k_xcyl, self.k_rc)

    @property
    def k_rs(self):
        """
        Shell-side (cell-free fluid layer) radial permeability [m2]. 
        """
        cyl_packing = self.cyl_packing
        k_rcyl = ((self.R_C**2)/(8*cyl_packing)) * (
                -math.log(cyl_packing) + (cyl_packing**2-1)/(cyl_packing**2+1))
        return max(k_rcyl, self.k_rc)

    @property
    def k_ths(self):
        """
        Shell-side (cell-free fluid layer) angular permeability [m2]. 
        """
        return self.k_rs
    
    @property
    def settling_v(self):
        """
        Settling velocity of suspended cells on shell-side [m/s] in cartesian
        coordinates. Velocity returned decomposed into i) z-component and 
        ii) y-component (no x-component).
        """
        Vsettling = 0.9*(((self.cell_rho-self.rho) * (self.cell_r*2)**2)
                         / (18*self.mu)) * -9.8
        return [Vsettling*a(self.angle) for a in [math.sin, math.cos]] 


def solve_pv(
        hfb,
        rev_x = 1,
        rev_r = 3,
        NTH = 12,
        max_iter=5000,
        tol = 1e-9
        ):
    """
    Solves for the pressure and velocity profiles for a given hollow fiber
    bioreactor object and grid.

    Parameters
    ----------
    hfb : hollow_fiber_bioreactor object
        Expects a hollow-fiber_bioreactor object.
    rev_x : int, default 1
        Defines grid based on NX = ceil(L/rev_x*rev_r*krogh cylinder diameter)
        where NX is number of points in grid in axial direction.
    rev_r : int, default 3
        Defines grid based on NR = ceil(R/rev_r*krogh cylinder diameter)
        where NR is number of points in grid in radial direction.
    NTH : int, default 12
        NTH is number of points in grid in theta direction. This needs to be an 
        even number so that a control volume face falls at theta=0 and theta=pi.
    max_iter : int, default 5000
        Max number of times to iteratively resolve lumen pressure profile 
        based on updated ecs pressure profile before exiting loop.
    tol : float, default 1e-9
        Infinity norm of lumen pressure profile residual tolerance.

    Returns
    -------
    solution : tuple 
        A tuple containing the following elements:
            1) A list containing the locations of the control volume centers
            (the axial meshgrid, radial meshgrid, and theta meshgrid).
            2) A Collections named tuple for the lumen.
            3) A Collections named tuple for the ecs.
        where each Collections named tuple contains the following:
            pressure : numpy.ndarray
                Pressure at the control volume centers [kPa]
            face_vel : list
                List of the decomposed superficial velocies at the 
                control volume faces (axial, radial, theta) [m/s] 
            center_vel : list 
                List of the decomposed superficial velocities at the control 
                volume centers (axial, radial, theta) [m/s]
    """
    def assemble_matrix(axial, bc_x, radial, bc_r, angular):
        """
        Assembles sparse matrices of discretized equations' coefficients.

        Parameters
        ----------
        axial : float
            Axial transport dimensionless coefficient.
        bc_x : list
            List of axial boundary conditions. Each boundary condition is a 
            list providing the index, type, and value.
        radial : float
            Radial transport dimensionless coefficient.
        bc_r : list
            List of radial boundary conditions. Each boundary condition is a 
            list providing the index, type, and value.
        angular : float
            Angular transport dimensionless coefficient.

        Returns
        -------
        A : sparse._dia.dia_matrix 
            Sparse matrix of coefficients.
        aP : numpy.ndarray
            Main diagonal of matrix.
        b : numpy.ndarray
            Constant (right hand side) vector.
        """
        # Axial coefficient diagonals
        aW = np.full(Pr.shape, -axial)  # Offset -1
        aE = aW.copy()  # Offset 1

        # Begin building main diagonal and right hand side vector
        aP = source - aW - aE
        b = np.zeros((NTH, NR, NX))

        # Axial boundary conditions
        for index, ttype, value in bc_x:
            if ttype == 'dirichlet':
                aP[index] = aP[index] + axial
                b[index] = 2*axial * value
            else:
                aP[index] = aP[index] - axial
                b[index] = axial * dx * value

        # No 'west' or 'east' neighbor at spatial domain boundary
        aW[:, :, 0] = 0
        aE[:, :, -1] = 0

        # Assemble tridiagonal matrix if no radial and theta component
        if not radial:
            A = sparse.diags([(aW/aP).ravel()[1:],
                              np.ones(NR*NX*NTH),
                              (aE/aP).ravel()[:-1]],
                             [-1, 0, 1])
            return A, aP.ravel(), b.ravel()

        # Radial coefficient diagonals
        aN = -radial * (Pr + 0.5*dr)/Pr
        aS = -radial * (Pr - 0.5*dr)/Pr

        # Outer radial boundary conditions
        for index, ttype, value in bc_r:
            if ttype == 'dirichlet':
                aP[index] = aP[index] - (2*aN[index])
                b[index] = b[index] - (2*aN[index]*value)
            else:
                b[index] = b[index] - aN[index]*dr*value

        # No 'north' neighbor at boundary (aS[0,:,:] already eq. 0)
        aN[:, -1, :] = 0

        # Angular coefficient diagonals  
        aT = -angular 
        aB = aT.copy()

        # Angular boundary conditions are symmetry at theta=0 and theta=pi
        aT[int(NTH/2)-1::int(NTH/2)] = 0
        aB[:-1:int(NTH/2)] = 0

        # Update main diagonal with radial and angular coefficients (axial 
        # already included)
        aP = aP - aN - aS - aT - aB

        A = sparse.diags([(aW/aP).ravel()[1:],
                          np.ones(NR*NX*NTH),
                          (aE/aP).ravel()[:-1],
                          (aN/aP).ravel()[:-NX],
                          (aS/aP).ravel()[NX:],
                          (aT/aP).ravel()[:-NX*NR],
                          (aB/aP).ravel()[NX*NR:]],
                         [-1, 0, 1, NX, -NX, NX*NR, -NX*NR])

        return A, aP.ravel(), b.ravel()

    # Domain Discretization (pressure grid staggered from velocity grid)
    NR = math.ceil(hfb.R / (hfb.R_S * 2 * rev_r))
    NX = math.ceil(hfb.L / (hfb.R_S * 2 * rev_r * rev_x))

    dr, dx, dth = hfb.R/NR, hfb.L/NX, 2*math.pi/NTH

    # Control volume faces (velocity)
    vr, vth, vx = np.meshgrid(np.linspace(0, hfb.R, NR+1)[1:],
                              np.linspace(0, 2*math.pi, NTH+1)[1:],
                              np.linspace(0, hfb.L, NX+1)[1:])

    # Control volume centers (pressure)
    Px, Pr, Pth = [a-b/2 for a, b in zip([vx, vr, vth], [dx, dr, dth])]

    # Discretized Pressure Equation Dimensionless Constants
    lum_x = hfb.k_xl / (dx**2)  # Lumen axial transport
    ecs_x = hfb.k_xs / (dx**2)  # ECS axial transport
    ecs_r = hfb.k_rs / (dr**2)  # ECS radial transport
    ecs_th = hfb.k_ths / ((Pr*dth)**2)  # ECS angular transport
    source = hfb.mu * hfb.LP * hfb.a_v  # Source (or sink ) term

    # ECS Port locations 
    port_th = 2  # number of symmetrical angular points in ECS port
    port_x = int(round(0.1*hfb.L/dx))  # number of axial points in ECS port
    up, down = [slice(None, port_x), slice(-port_x, None)]
    top, bttm = [(slice(None, port_th), slice(-port_th, None)),
                 slice(int(NTH/2)-port_th, int(NTH/2)+port_th)]

    # Boundary Conditions
    lum_bc_x = [[(slice(None), slice(None), 0), *hfb.lum_in],
                [(slice(None), slice(None), -1), *hfb.lum_out]]
    ecs_bc_x = [[(slice(None), slice(None), 0), 'neumann', 0],
                [(slice(None), slice(None), -1), 'neumann', 0]]
    ecs_bc_r = [[(top[0], -1, up), *hfb.ecs_upstreamtop],
                [(top[1], -1, up), *hfb.ecs_upstreamtop],
                [(bttm, -1, up), *hfb.ecs_upstreambttm],
                [(top[0], -1, down), *hfb.ecs_downstreamtop],
                [(top[1], -1, down), *hfb.ecs_downstreamtop],
                [(bttm, -1, down), *hfb.ecs_downstreambttm]]

    # Assemble Tridiagonal Matrix A and vector b for Lumen Pressure
    A_lum, aP_lum, b_lum0 = assemble_matrix(lum_x, lum_bc_x, None, None, None)

    # Assemble 7-diagonal Matrix A and vector b for ECS Pressure
    A_ecs, aP_ecs, b_ecs0 = assemble_matrix(ecs_x, ecs_bc_x, ecs_r, ecs_bc_r,
                                            ecs_th)

    # Assemble Initial Guess Vector b for Lumen
    b_lum = (b_lum0 + 100*source) / aP_lum

    # Solve Coupled PDEs for Pressure Profiles
    iteration = 0
    while True:
        # Solve for Lumen Pressure
        pL = sparse.linalg.spsolve(A_lum.tocsr(), b_lum)
        b_ecs = (b_ecs0 + source*pL) / aP_ecs

        # Solve for ECS Pressure
        pS = sparse.linalg.spsolve(A_ecs.tocsr(), b_ecs)
        b_lum_prev = b_lum.copy()
        b_lum = (b_lum0 + source*pS) / aP_lum

        # Calculate Norms of Residual Vectors
        norm_diff = np.linalg.norm(b_lum-b_lum_prev, np.inf)

        iteration += 1
        print(norm_diff)

        if norm_diff < tol:
            break

        if iteration > max_iter:
            print('Soln not found in {} iterations'.format(max_iter))
            break

    # Pressure [kPa] at the control volume centers
    pL, pS = [a.reshape(NTH, NR, NX) for a in [pL, pS]]

    # Axial superficial velocities at control volume faces [m/s]
    vL_xf, vS_xf = [np.zeros((NTH, NR, NX+1)) for foo in range(2)]
    vL_xf[:,:,1:-1] = (-hfb.k_xl/(hfb.mu*1e-3))*(pL[:,:,1:]-pL[:,:,0:-1])/dx
    vS_xf[:,:,1:-1] = (-hfb.k_xs/(hfb.mu*1e-3))*(pS[:,:,1:]-pS[:,:,0:-1])/dx
    
    for index, ttype, value in lum_bc_x:
        if ttype == 'dirichlet':
            value = np.abs(value - pL[index]) / (0.5*dx) # left-->right flow
        vL_xf[index] = (hfb.k_xl/(hfb.mu*1e-3)) * value
            
    # Radial superficial velocities at control volume faces [m/s]
    vS_rf = np.zeros((NTH, NR+1, NX))
    vS_rf[:,1:-1,:] = (-hfb.k_rs/(hfb.mu*1e-3))*(pS[:,1:,:]-pS[:,0:-1,:])/dr
    
    for index, ttype, value in ecs_bc_r:  # any non-Neumann BC of 0 at r=R
        if not(ttype=='neumann' and value==0):
            if ttype=='dirichlet':
                value = (value - pS[index]) / (0.5*dr)
            vS_rf[index] = (-hfb.k_rs/(hfb.mu*1e-3)) * value
    
    # Theta superficial velocities at control volume faces [m/s]
    vS_thf = (-hfb.k_rs/(hfb.mu*1e-3))*(np.roll(pS, (-1,0,0), axis=0)
                                        - pS)/(Pr*dth)
    
    # Superficial velocities at control volume centers [m/s] based on linear
    # interpolation of face values 
    vL_x = (vL_xf[:,:,1:] + vL_xf[:,:,:-1]) / 2
    vS_x = (vS_xf[:,:,1:] + vS_xf[:,:,:-1]) / 2
    vS_r = (vS_rf[:,1:,:] + vS_rf[:,:-1,:]) / 2
    vS_th = (vS_thf + np.roll(vS_thf, (1,0,0), axis=0)) / 2
    
    soln = namedtuple('soln', ['pressure', 'face_vel', 'center_vel'])

    return ([Px, Pr, Pth], soln(pL, [vL_xf], [vL_x]), 
            soln(pS, [vS_xf, vS_rf, vS_thf], [vS_x, vS_r, vS_th]),
            {'up': up, 'down': down, 'top': top, 'bttm': bttm})


def vol_bal_check(hfb, domain, lum, ecs):
    """
    Checks if solution returned by solve_pv satisfies volume balance.

    Parameters
    ----------
    hfb : hollow_fiber_bioreactor
        Hollow-fiber bioreactor object (see class definition).
    domain : list
        List returned by solve_pv function containing the locations of the
        control volume centers (the axial meshgrid, radial meshgrid, and 
        theta meshgrid).
    lum : soln collections named tuple
        Collections named tuple returned by solve_pv containing the lumen
        pressure, control volume face superficial velocities, and control 
        volume center superficial velocities (axial, radial, theta) [m/s]
    ecs : soln collections named tuple
        Collections named tuple returned by solve_pv containing the ECS
        pressure, control volume face superficial velocities, and control 
        volume center superficial velocities (axial, radial, theta) [m/s]

    Returns
    -------
    axial : numpy.ndarray
        Axial volume balance.  
    radial : numpy.ndarray
        Radial volume balance.
    angular : numpy.ndarray
        Angular volume balance.
    source : numpy.ndarray
        Transmembrane source (lum to ECS) or sink (ECS to lum). Source is
        positive when source and negative when sink.
    overall : numpy.ndarray
        Overall volume balance (Out - In - Source = 0).

    """
    x, r, th = domain
    dx, dr, dth = [a[1] - a[0] for a in [x[0, 0, :], r[0, :, 0], th[:, 0, 0]]]
    vx, vr, vth = ecs.face_vel
    
    # Out - In - Source = 0 (where 'source' is pos when in and neg when sink)
    axial = (vx[:,:,1:] - vx[:,:,:-1]) / dx
    radial = (vr[:,1:,:]*(r+0.5*dr) - vr[:,:-1,:]*(r-0.5*dr)) / (r*dr)
    angular = (vth - np.roll(vth, (1,0,0), axis=0)) / (r*dth)
    source = hfb.LP * hfb.a_v * (lum.pressure - ecs.pressure)*1000
    overall = axial + radial + angular - source

    return axial, radial, angular, source, overall


def concentration(hfb, domain, lum, ecs, doubling_t=24.0,
                  scale=1e-6, steady_state=True,
                  dt=1.0, total_t=24.0, C_init=0
                  ):
    """
    Solves for the suspended cell concentration profile for a given hollow 
    fiber bioreactor object, grid, velocity profile, and growth parameters.

    Parameters
    ----------
    hfb : hollow_fiber_bioreactor
        Hollow-fiber bioreactor object (see class definition).
    domain : list
        List returned by solve_pv function containing the locations of the
        control volume centers (the axial meshgrid, radial meshgrid, and 
        theta meshgrid).
    lum : soln collections named tuple
        Collections named tuple returned by solve_pv containing the lumen
        pressure, control volume face superficial velocities, and control 
        volume center superficial velocities (axial, radial, theta) [m/s]
    ecs : soln collections named tuple
        Collections named tuple returned by solve_pv containing the ECS
        pressure, control volume face superficial velocities, and control 
        volume center superficial velocities (axial, radial, theta) [m/s]
    doubling_t : float, optional
        Cell doubling time [h]. The default is 24.
    scale : float, optional
        Scales the concentration of suspended cells per mL of total volume. 
        The default is 1e-6.
    steady_state : bool, optional
        Solve for steady-state suspended cell concentration, exiting iterations
        once steady-state condition is met. If false, harvest will be modeled 
        for the total time indicated by total_t. The default is True.
    dt : float, optional
        Time step size [h] if unsteady harvest is modeled. The default is 1.0.
    total_t : float, optional
        Total unsteady harvest time to model. The default is 24h.
    C_init : float or numpy.ndarray, optional
        Initial suspended cell concentration. The default is 0.

    Returns
    -------
    conc : list
        List storing the concentration profile numpy.ndarrays (concentration
        at each control volume center) obtained either at steady-state or at
        each time step (starting with t=0 at index=0). 
    """
    x, r, th = domain
    NTH, NR, NX = x.shape
    dx, dr, dth = [a[1] - a[0] for a in [x[0, 0, :], r[0, :, 0], th[:, 0, 0]]]
    source = scale*hfb.cell_dens*math.log(2)/(doubling_t)  #[1e6 cells/mL*h]
    
    # Gravity (interstitial settling velocity) [m/s] in cartesian coordinates
    # converted to superficial [m/h] and then converted to cylindrical.
    gz, gy = [3600 * a * (1-hfb.cyl_packing) for a in hfb.settling_v]
    gr = gy * np.sin(th + 0.5*math.pi) 
    gth = gy * np.cos(th + 0.5*dth + 0.5*math.pi)
    
    # Superficial velocities at control vol faces [m/s]--> [m/h] combined with
    # settling velocities [m/h].
    vx, vr, vth = [3600*foo for foo in ecs.face_vel]
    vx[:, :, 1:-1] = vx[:, :, 1:-1] + gz
    vr[:, :-1, :] = vr[:, :-1, :] + gr
    vth = vth + gth

    # Coefficients [1/h]
    aE = vx[:, :, 1:] / dx
    aW = -vx[:, :, :-1] / dx
    aN = vr[:, 1:, :] * ((r + 0.5*dr) / (r*dr))
    aS = -vr[:, :-1, :] * ((r - 0.5*dr) / (r*dr))
    aT = vth / (r*dth)
    aB = -np.roll(vth, (1,0,0), axis=0) / (r*dth)

    # At r=R, aN=0 where vr < 0 (flow into HFB through ports is cell free)
    aN[:, -1, :][aN[:, -1, :] < 0] = 0

    aP = np.zeros_like(x)
    b = np.full(aP.shape, source)
     
    # Upwind scheme
    for a in [aW, aE, aN, aS, aB, aT]:
        aP += np.maximum(a, 0)
        a[a > 0] = 0
    
    A = sparse.diags([(aW/aP).ravel()[1:],
                      np.ones(NR*NX*NTH),
                      (aE/aP).ravel()[:-1],
                      (aN/aP).ravel()[:-NX],
                      (aS/aP).ravel()[NX:],
                      (aT/aP).ravel()[:-NX*NR],
                      (aB/aP).ravel()[NX*NR:]],
                     [-1, 0, 1, NX, -NX, NX*NR, -NX*NR])

    # Solve for steady-state profile [1e6 cells/mL bioreactor volume]
    if steady_state:
        C = np.reshape(sparse.linalg.spsolve(A.tocsr(), (b/aP).ravel()),
                       x.shape)

        # Cell harvest [10^6 cells harvested in 'dt' time increment]
        harvest = np.sum(np.where(vr[:, -1, :]>0,
                                  (dt * vr[:, -1, :]*hfb.R*dth*dx 
                                   * 1e6 * C[:, -1, :]),
                                  0))

        return [C]

    else:
        conc = [C_init]
        harv = [0]
        iteration = 0
        aP += 1/dt
        b += C_init/dt

        while iteration<=int(total_t/(dt)):

            C = np.reshape(sparse.linalg.spsolve(A.tocsr(), (b/aP).ravel()),
                           x.shape)
            conc.append(C)
            b = source + C/dt

            # Cell harvest [10^6 cells harvested in 'dt' time increment]
            harvest = np.sum(np.where(vr[:, -1, :]>0,
                                      (dt * vr[:, -1, :]*hfb.R*dth*dx 
                                       * 1e6 * C[:, -1, :]),
                                      0))

            harv.append(harvest)
            iteration += 1

        accumulation = np.sum(C * 1e6 * r*dr*dth*dx)
        total = (iteration)*dt * source * 1e6 * (hfb.R**2*math.pi*hfb.L)

        print(harv, np.sum(harv), accumulation, total, total-accumulation,
              [np.sum(foo*1e6*r*dr*dth*dx) for foo in conc], iteration)

        return conc


def visual(hfb, domain, data, label, lab_format='%.0f'):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, r, th = domain
    xx, yy = [1000*r*foo(th+0.5*math.pi) for foo in [np.cos, np.sin]]

    sc = ax.scatter(100*x, xx, yy, c=data, cmap='viridis_r', marker='o', s=1)
    colorbar = plt.colorbar(sc, shrink=0.8, pad=0.13,
                            ticks=np.linspace(np.min(data),
                                              np.max(data), 5))
    colorbar.set_label(label)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter(lab_format))

    ax.view_init(elev=20, azim=-20) # default 30, -60; (20, -20)

    ax.set_xticks(np.linspace(0, 100*hfb.L, 5))
    ax.set_xlim(0, 100*hfb.L)
    ax.set_xlabel('Axial Distance [cm]')
    ax.set_ylabel('X [mm]')
    ax.set_zlabel('Y [mm]')
    plt.tight_layout()

    return


def visual_2D(x, y, z, title):
    fig, ax = plt.subplots(figsize=(8,6), dpi=200)

    im = ax.contourf(x, y, np.round(z, 2), 20, cmap='viridis_r')
    ax.set_xlabel('Axial Distance [cm]')
    ax.set_ylabel('Radial Distance [mm]')
    ax.set_xticks(np.linspace(*ax.get_xlim(), 5), 
                  [f"{xi:.1f}" for xi in np.linspace(0, ax.get_xlim()[1]+x[0], 5)])
    ax.set_yticks(np.linspace(*ax.get_ylim(), 5), 
                  [f"{xi:.1f}" for xi in np.linspace(0, ax.get_ylim()[1]+y[0], 5)])
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    return