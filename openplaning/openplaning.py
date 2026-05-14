import numpy as np
from scipy import interpolate, signal
from scipy.special import gamma
import ndmath
import warnings
from openplaning.tables_data import (
    RAW_02_DATA, RAW_04_DATA, RAW_06_DATA,
    V_02_DATA, V_04_DATA,
    RAW_V_02_DATA, RAW_V_04_DATA, RAW_V_06_DATA
)

class PlaningBoat():
    """Prismatic planing craft
    
    Attributes:
        speed (float): Speed (m/s). It is an input to :class:`PlaningBoat`. 
        weight (float): Weight (N). It is an input to :class:`PlaningBoat`.
        beam (float): Beam (m). It is an input to :class:`PlaningBoat`.
        lcg (float): Longitudinal center of gravity, measured from the stern (m). It is an input to :class:`PlaningBoat`.
        vcg (float): Vertical center of gravity, measured from the keel (m). It is an input to :class:`PlaningBoat`.
        r_g (float): Radius of gyration (m). It is an input to :class:`PlaningBoat`.
        beta (float): Deadrise (deg). It is an input to :class:`PlaningBoat`.
        epsilon (float): Thrust angle w.r.t. keel, CCW with body-fixed origin at 9 o'clock (deg). It is an input to :class:`PlaningBoat`.
        vT (float): Thrust vertical distance, measured from keel, and positive up (m). It is an input to :class:`PlaningBoat`.
        lT (float): Thrust horizontal distance, measured from stern, and positive forward (m). It is an input to :class:`PlaningBoat`.
        loa (float): Vessel LOA for seaway behavior estimates (m). Defaults to None. It is an input to :class:`PlaningBoat`.
        H_sig (float): Significant wave heigth in an irregular sea state (m). Defaults to None. It is an input to :class:`PlaningBoat`.
        ahr (float): Average hull roughness (m). Defaults to 150*10**-6. It is an input to :class:`PlaningBoat`.
        LD_change (float): Roughness induced change of hull lift to change of hull drag ratio (dimensionless). Defaults to None, but ITTC '78 approximates a value of -1.1 for propellers. It is an input to :class:`PlaningBoat`.
        Lf (float): Flap chord (m). Defaults to 0. It is an input to :class:`PlaningBoat`.
        sigma (float): Flap span-beam ratio (dimensionless). Defaults to 0. It is an input to :class:`PlaningBoat`.
        delta (float): Flap deflection (deg). Defaults to 0. It is an input to :class:`PlaningBoat`.
        l_air (float): Distance from stern to center of air pressure (m). Defaults to 0. It is an input to :class:`PlaningBoat`.
        h_air (float): Height from keel to top of square which bounds the air-drag-inducing area (m). Defaults to 0. It is an input to :class:`PlaningBoat`.
        b_air (float): Transverse width of square which bounds the air-drag-inducing area (m). Defaults to 0. It is an input to :class:`PlaningBoat`.
        C_shape (float): Area coefficient for air-drag-inducing area (dimensionless). C_shape = 1 means the air drag reference area is h_air*b_air. Defaults to 0. It is an input to :class:`PlaningBoat`.
        C_D (float): Air drag coefficient (dimensionless). Defaults to 0.7. It is an input to :class:`PlaningBoat`.
        rho (float): Water density (kg/m^3). Defaults to 1025.87. It is an input to :class:`PlaningBoat`.
        nu (float): Water kinematic viscosity (m^2/s). Defaults to 1.19*10**-6. It is an input to :class:`PlaningBoat`.
        rho_air (float): Air density (kg/m^3). Defaults to 1.225. It is an input to :class:`PlaningBoat`.
        g (float): Gravitational acceleration (m/s^2). Defaults to 9.8066. It is an input to :class:`PlaningBoat`.
        z_wl (float): Vertical distance of center of gravity to the calm water line (m). Defaults to 0. It is an input to :class:`PlaningBoat`, but modified when running :meth:`get_steady_trim`.
        tau (float): Trim angle (deg). Defaults to 5. It is an input to :class:`PlaningBoat`, but modified when running :meth:`get_steady_trim`.
        eta_3 (float): Additional heave (m). Initiates to 0.
        eta_5 (float): Additional trim (deg). Initiates to zero.
        wetted_lengths_type (int): 1 = Use Faltinsen 2005 wave rise approximation, 2 = Use Savitsky's '64 approach, 3 = Use Savitsky's '76 approach. Defaults to 1. It is an input to :class:`PlaningBoat`.
        z_max_type (int): 1 = Uses 3rd order polynomial fit, 2 = Uses cubic interpolation from table. This is only used if wetted_lenghts_type == 1. Defaults to 1. It is an input to :class:`PlaningBoat`.
        L_K (float): Keel wetted length (m). It is updated when running :meth:`get_geo_lengths`.
        L_C (float): Chine wetted length (m). It is updated when running :meth:`get_geo_lengths`.
        L_C2 (float): Side chine wetted length with reattached flow (m). It is updated when running :meth:`get_geo_lengths`.
        wetted_bottom_area (float): Bottom wetted surface area (m^2). It is updated when running :meth:`get_geo_lengths`.
        lambda_W (float): Mean wetted-length to beam ratio, (L_K+L_C)/(2*beam) (dimensionless). It is updated when running :meth:`get_geo_lengths`.
        x_s (float): Distance from keel/water-line intersection to start of wetted chine (m). It is updated when running :meth:`get_geo_lengths`.
        alpha (float): Angle between spray line and keel, projected to plan view (deg). It is updated when running :meth:`get_geo_lengths`.
        z_max (float): Maximum pressure coordinate coefficient, z_max/Ut (dimensionless). It is updated when running :meth:`get_geo_lengths`.
        T (float): Transom draft (m). It is updated when running :meth:`get_geo_lengths`.
        lcp (float): Longitudinal center of pressure, measured from the stern (m). It is updated when running :meth:`get_forces`.
        roughness_penalty_type (int): 1 = Use Mosaad's '86 regression, 2 = Use Townsin's '84 regression. Defaults to 1. It is an input to :class:`PlaningBoat`.
        C_Lbeta (float): Lift coefficient with deadrise. It is updated when running :meth:`get_forces`.
        deltaC_L (float): Change in hydrodynamic lift coefficient due to roughness, excluding lift change due to roughness. It is updated when running :meth:`get_forces`.
        hydrodynamic_force ((3,) ndarray): Hydrodynamic force (N, N, N*m). [F_x, F_z, M_cg] with x, y, rot directions in intertial coordinates. It is updated when running :meth:`get_forces`.
        bottom_fluid_speed (float): Mean bottom fluid speed (m/s). It is updated when running :meth:`get_forces`.
        C_f (float): Friction coefficient, smooth case. It is updated when running :meth:`get_forces`.
        deltaC_f (float): Change in friction coefficient due to roughness. It is updated when running :meth:`get_forces`.
        skin_friction ((3,) ndarray): Skin friction force (N, N, N*m). [F_x, F_z, M_cg]. It is updated when running :meth:`get_forces`.
        lift_change ((3,) ndarray): Lift change due to roughness (N, N, N*m). [F_x, F_z, M_cg]. It is updated when running :meth:`get_forces`.
        air_resistance ((3,) ndarray): Air resistance force (N, N, N*m). [F_x, F_z, M_cg]. It is updated when running :meth:`get_forces`.
        flap_force ((3,) ndarray): Flap resultant force (N, N, N*m). [F_x, F_z, M_cg]. It is updated when running :meth:`get_forces`.
        thrust_force ((3,) ndarray): Thrust resultant force (N, N, N*m). [F_x, F_z, M_cg]. It is updated when running :meth:`get_forces`.
        net_force ((3,) ndarray): Net force (N, N, N*m). [F_x, F_z, M_cg]. It is updated when running :meth:`get_forces`.
        mass_matrix ((2, 2) ndarray): Mass coefficients matrix. [[A_33 (kg), A_35 (kg*m/rad)], [A_53 (kg*m), A_55 (kg*m^2/rad)]]. It is updated when running :meth:`get_eom_matrices`.
        damping_matrix ((2, 2) ndarray): Damping coefficients matrix. [[B_33 (kg/s), B_35 (kg*m/(s*rad))], [B_53 (kg*m/s), B_55 (kg*m**2/(s*rad))]]. It is updated when running :meth:`get_eom_matrices`.
        restoring_matrix ((2, 2) ndarray): Restoring coefficients matrix. [[C_33 (N/m), C_35 (N/rad)], [C_53 (N), C_55 (N*m/rad)]]. It is updated when running :meth:`get_eom_matrices`.
        porpoising (list): [[eigenvalue result (bool), est. pitch settling time (s)], [Savitsky chart result (bool), critical trim angle (deg)]].  It is updated when running :meth:`check_porpoising`.
        seaway_drag_type (int): 1 = Use Savitsky's '76 approximation, 2 = Use Fridsma's '71 designs charts. Defaults to 1. It is an input to :class:`PlaningBoat`.
        avg_impact_acc ((2,) ndarray): Average impact acceleration at center of gravity and bow (g's). [n_cg, n_bow]. It is updated when running :meth:`get_seaway_behavior`.
        R_AW (float): Added resistance in waves (N). It is updated when running :meth:`get_seaway_behavior`.
    """
    
    def __init__(self, speed, weight, beam, lcg, vcg, r_g, beta, epsilon, vT, lT, loa=None, H_sig=None, ahr=150e-6, LD_change=None, Lf=0, sigma=0, delta=0, l_air=0, h_air=0, b_air=0, C_shape=0, C_D=0.7, z_wl=0, tau=5, rho=1025.87, nu=1.19e-6, rho_air=1.225, g=9.8066, wetted_lengths_type=1, z_max_type=1, roughness_penalty_type=1, seaway_drag_type=1):
        """Initialize attributes for PlaningBoat
        
        Args:
            speed (float): Speed (m/s).
            weight (float): Weidght (N).
            beam (float): Beam (m).
            lcg (float): Longitudinal center of gravity, measured from the stern (m).
            vcg (float): Vertical center of gravity, measured from the keel (m).
            r_g (float): Radius of gyration (m).
            beta (float): Deadrise (deg).
            epsilon (float): Thrust angle w.r.t. keel, CCW with body-fixed origin at 9 o'clock (deg).
            vT (float): Thrust vertical distance, measured from keel, and positive up (m).
            lT (float): Thrust horizontal distance, measured from stern, and positive forward (m).
            loa (float, optional): Vessel LOA for seaway behavior estimates (m). Defaults to None.
            H_sig (float, optional): Significant wave heigth in an irregular sea state (m). Defaults to None.
            ahr (float, optional): Average hull roughness (m). Defaults to 150*10**-6.
            LD_change (float, optional): Roughness induced change of hull lift to change of hull drag ratio (dimensionless). Defaults to None (ITTC '78 estimates -1.1).
            Lf (float, optional): Flap chord (m). Defaults to 0.
            sigma (float, optional): Flap span-beam ratio (dimensionless). Defaults to 0.
            delta (float, optional): Flap deflection (deg). Defaults to 0.
            l_air (float, optional): Distance from stern to center of air pressure (m). Defaults to 0.
            h_air (float, optional): Height from keel to top of square which bounds the air-drag-inducing area (m). Defaults to 0.
            b_air (float, optional): Transverse width of square which bounds the air-drag-inducing area (m). Defaults to 0.
            C_shape (float, optional): Area coefficient for air-drag-inducing area (dimensionless). C_shape = 1 means the air drag reference area is h_air*b_air. Defaults to 0.
            C_D (float, optional): Air drag coefficient (dimensionless). Defaults to 0.7.
            z_wl (float, optional): Vertical distance of center of gravity to the calm water line (m). Defaults to 0.
            tau (float, optional): Trim angle (deg). Defaults to 5.
            rho (float, optional): Water density (kg/m^3). Defaults to 1025.87.
            nu (float, optional): Water kinematic viscosity (m^2/s). Defaults to 1.19*10**-6.
            rho_air (float, optional): Air density (kg/m^3). Defaults to 1.225.
            g (float, optional): Gravitational acceleration (m/s^2). Defaults to 9.8066.
            wetted_lengths_type (int, optional): 1 = Use Faltinsen 2005 wave rise approximation, 2 = Use Savitsky's '64 approach, 3 = Use Savitsky's '76 approach. Defaults to 1.
            z_max_type (int, optional): 1 = Uses 3rd order polynomial fit, 2 = Uses cubic interpolation from table. This is only used if wetted_lenghts_type == 1. Defaults to 1.
            roughness_penalty_type (int, optional): 1 = Use Mosaad's '86 regression, 2 = Use Townsin's '84 regression. Defaults to 1.
            seaway_drag_type (int, optional): 1 = Use Savitsky's '76 approximation, 2 = Use Fridsma's '71 designs charts. Defaults to 1.
        """
        self.speed = speed
        self.weight = weight
        self.beam = beam
        self.lcg = lcg 
        self.vcg = vcg 
        self.r_g = r_g 
        self.beta = beta
        self.epsilon = epsilon 
        self.vT = vT 
        self.lT = lT
        self.loa = loa
        self.H_sig = H_sig
        self.ahr = ahr
        self.LD_change = LD_change
        self.Lf = Lf
        self.sigma = sigma
        self.delta = delta
        self.l_air = l_air
        self.h_air = h_air
        self.b_air= b_air
        self.C_shape = C_shape
        self.z_wl = z_wl
        self.tau = tau
        self.eta_3 = 0
        self.eta_5 = 0
        self.rho = rho
        self.nu = nu
        self.rho_air = rho_air
        self.C_D = C_D
        self.g = g
        
        self.gravity_force = np.array([0, -self.weight, 0])
        
        self.wetted_lengths_type = wetted_lengths_type
        self.z_max_type = z_max_type

        self.roughness_penalty_type = roughness_penalty_type

        self.seaway_drag_type = seaway_drag_type
        
    def print_description(self, sigFigs=7, runAllFunctions=True):
        """Returns a formatted description of the vessel.
        
        Args:
            sigFigs (int, optional): Number of significant figures to display. Defaults to 7.
            runAllFunctions (bool, optional): Runs all functions with default values before printing results. Defaults to True.
        """
        if runAllFunctions:
            self.get_geo_lengths()
            self.get_forces(runGeoLengths=False)
            self.get_eom_matrices(runGeoLengths=False)
            self.get_seaway_behavior()
            self.check_porpoising()
        
        volume = self.weight/(self.g*self.rho)
        
        table = [
            ['---VESSEL---'],
            ['Speed', self.speed, 'm/s'],
            ['V_k', self.speed*1.944, 'knot'],
            ['Fn (beam)', self.speed/np.sqrt(self.g*self.beam), ''],
            ['Fn (volume)', self.speed/np.sqrt(self.g*(volume)**(1/3)), ''],
            ['V_m', self.bottom_fluid_speed, 'm/s, mean bottom fluid speed'],
            ['Rn', self.bottom_fluid_speed * self.lambda_W * self.beam / self.nu, 'based on V_m and mean wetted-length'],
            [''],
            ['Weight', self.weight, 'N'],
            ['Mass', self.weight/self.g, 'kg'],
            ['Volume', volume, 'm\u00B3'],
            ['Beam', self.beam, 'm'],
            ['LCG', self.lcg, 'm from stern'],
            ['VCG', self.vcg, 'm from keel'],
            ['R_g', self.r_g, 'm'],
            ['Deadrise', self.beta, 'deg'], #'\N{greek small letter beta}'
            [''],
            ['LOA', self.loa, 'm'],
            ['AHR', self.ahr*10**6, '10\u207b\u2076m, average hull roughness'],
            [''],
            ['---ATTITUDE---'],
            ['z_wl', self.z_wl, 'm, vertical distance of center of gravity to the calm water line'],
            ['tau', self.tau, 'deg, trim angle'],
            ['\u03B7\u2083', self.eta_3, 'deg, additional heave'],
            ['\u03B7\u2085', self.eta_5, 'deg, additional trim'],
            [''],
            ['---PROPULSION---'],
            ['Thrust angle', self.epsilon, 'deg w.r.t. keel (CCW with body-fixed origin at 9 o\'clock)'],
            ['LCT', self.lT, 'm from stern, positive forward'],
            ['VCT', self.vT, 'm from keel, positive up'],
            [''],
            ['---FLAP---'],
            ['Chord', self.Lf, 'm'],
            ['Span/Beam', self.sigma, ''],
            ['Angle', self.delta, 'deg w.r.t. keel (CCW with body-fixed origin at 9 o\'clock)'],
            [''],
            ['---AIR DRAG---'],
            ['l_air', self.l_air, 'm, distance from stern to center of air pressure'],
            ['h_air', self.h_air, 'm, height from keel to top of square which bounds the air-drag-inducing shape'],
            ['b_air', self.b_air, 'm, transverse width of square which bounds the air-drag-inducing shape'],
            ['C_shape', self.C_shape, 'area coefficient for air-drag-inducing shape. C_shape = 1 means the air drag reference area is h_air*b_air'],
            ['C_D', self.C_D, 'air drag coefficient'],
            [''],
            ['---ENVIRONMENT---'],
            ['\u03C1', self.rho, 'kg/m\u00B3, water density'],
            ['\u03BD', self.nu, 'm\u00B2/s, water kinematic viscosity'],
            ['\u03C1_air', self.rho_air, 'kg/m\u00B3, air density'],
            ['g', self.g, 'm/s\u00B2, gravitational acceleration'],
            [''],
            ['---WETTED LENGTH OPTIONS---'],
            ['wetted_lengths_type', self.wetted_lengths_type, '(1 = Use Faltinsen 2005 wave rise approximation, 2 = Use Savitsky\'s \'64 approach, 3 = Use Savitsky\'s \'76 approach)'],
            ['z_max_type', self.z_max_type, '(1 = Uses 3rd order polynomial fit (faster, recommended), 2 = Use cubic interpolation)'],
            [''],
            ['---RUNNING GEOMETRY---'],
            ['L_K', self.L_K, 'm, keel wetted length'],
            ['L_C', self.L_C, 'm, chine wetted length'],
            ['L_C2', self.L_C2, 'm, side chine wetted length'],
            ['\u03BB', self.lambda_W, 'mean wetted-length to beam ratio (L_K+L_C)/(2*beam)'],
            ['x_s', self.x_s, 'm, distance from keel/water-line intersection to start of wetted chine'],
            ['z_max', self.z_max, 'maximum pressure coordinate coefficient (z_max/Ut)'],
            ['alpha', self.alpha, 'deg, spray line angle w.r.t. keel in plan view'],
            ['LCP', self.lcp, 'm, longitudinal center of pressure from stern'],
            ['T', self.T, 'm, draft of keel at transom'],
            ['wetted_bottom_area', self.wetted_bottom_area, 'm\u00B2, bottom wetted surface area'],
            [''],
            ['---ROUGHNESS DRAG PENALTY---'],
            ['roughness_penalty_type', self.roughness_penalty_type, '(1 = Use Mosaad\'s \'86 regression, 2 = Use Townsin\'s \'84 regression)'],
            ['\u0394C_f', self.deltaC_f*10**3, '10\u207b\u00b3 change in friction coefficient'],
            ['\u0394L/\u0394D', self.LD_change, 'roughness induced change of hull lift to change of hull drag ratio'],
            ['\u0394C_L', self.deltaC_L*10**3, '10\u207b\u00b3 change in lift coefficient'],
            [''],
            ['---FORCES [F_x (N, +aft), F_z (N, +up), M_cg (N*m, +pitch up)]---'],
            ['Hydrodynamic Force', self.hydrodynamic_force, ''],
            ['Skin Friction', self.skin_friction, ''],
            ['Roughness Lift Change', self.lift_change, ''],
            ['Air Resistance', self.air_resistance, ''],
            ['Flap Force', self.flap_force, ''],
            ['Net Force', self.net_force, ''],
            ['Resultant Thrust', self.thrust_force, ''],
            ['---THURST & POWER---'],
            ['Thrust Magnitude', np.sqrt(self.thrust_force[0]**2+self.thrust_force[1]**2), 'N'],
            ['Effective Thrust', -self.thrust_force[0], 'N'],
            ['Eff. Power', -self.thrust_force[0]*self.speed/1000, 'kW'],
            ['Eff. Horsepower', -self.thrust_force[0]*self.speed/1000/0.7457, 'hp'],
            [''],
            ['---EOM MATRICES---'],
            ['Mass matrix, [kg, kg*m/rad; kg*m, kg*m\u00B2/rad]', self.mass_matrix, ''],
            ['Damping matrix, [kg/s, kg*m/(s*rad); kg*m/s, kg*m\u00B2/(s*rad)]', self.damping_matrix, ''],
            ['Restoring matrix, [N/m, N/rad; N, N*m/rad]', self.restoring_matrix, ''],
            ['---PORPOISING---'],
            ['[[Eigenvalue check result, Est. pitch settling time (s)],\n [Savitsky chart result, Critical trim angle (deg)]]', np.array(self.porpoising), ''],
            ['---BEHAVIOR IN WAVES---'],
            ['H_sig', self.H_sig, 'm, significant wave heigth'],
            ['R_AW', self.R_AW, 'N, added resistance in waves'],
            ['Average impact acceleration [n_cg, n_bow] (g\'s)', self.avg_impact_acc, ''],
        ]            
        
        cLens=[16,0,0] #Min spacing for columns
        for row in table:
            if len(row)==3:
                if row[1] is None:
                    print('{desc:<{cL0}} {val:<{cL1}} {unit:<{cL2}}'.format(desc=row[0], val='None', unit=row[2], cL0=cLens[0], cL1=cLens[1], cL2=cLens[2]))
                elif isinstance(row[1], (list,np.ndarray)):
                    print(row[0]+' =')
                    with np.printoptions(formatter={'float': f'{{:.{sigFigs}g}}'.format}):
                        print(row[1])
                    print(row[2])
                else:
                    print('{desc:<{cL0}} {val:<{cL1}.{sNum}g} {unit:<{cL2}}'.format(desc=row[0], val=row[1], unit=row[2], cL0=cLens[0], cL1=cLens[1], cL2=cLens[2], sNum=sigFigs))
            else:
                print(row[0])
        
    def get_geo_lengths(self):
        """This function outputs the geometric lengths. 
        
        Adds/updates the following attributes:

        - :attr:`L_K`
        - :attr:`L_C`
        - :attr:`lambda_W`
        - :attr:`x_s`
        - :attr:`z_max`
        """
        U = self.speed
        b = self.beam
        lcg = self.lcg
        vcg = self.vcg
        z_wl = self.z_wl
        tau = self.tau
        beta = self.beta
        eta_3 = self.eta_3
        eta_5 = self.eta_5
        pi = np.pi
        wetted_lengths_type = self.wetted_lengths_type
        z_max_type = self.z_max_type
        g = self.g
        
        #Keel wetted length, Eq. 9.50 of Faltinsen 2005, page 367
        L_K = lcg + vcg / np.tan(pi/180*(tau + eta_5)) - (z_wl + eta_3) / np.sin(pi/180*(tau + eta_5))
        if L_K < 0:
            L_K = 0
        
        if wetted_lengths_type == 1:
            #z_max/Vt coefficient, Table 8.3 of Faltinsen 2005, page 303---------------
            beta_table = [4, 7.5, 10, 15, 20, 25, 30, 40]
            z_max_table = [0.5695, 0.5623, 0.5556, 0.5361, 0.5087, 0.4709, 0.4243, 0.2866]

            #Extrapolation warning
            if beta < beta_table[0] or beta > beta_table[-1]:
                warnings.warn('Deadrise ({0:.3f}) outside the interpolation range of 4-40 deg (Table 8.3 of Faltinsen 2005). Extrapolated values might be inaccurate.'.format(beta), stacklevel=2)

            if z_max_type == 1:
                z_max = np.polyval([-2.100644618790201e-006, -6.815747611588763e-005, -1.130563334939335e-003, 5.754510457848798e-001], beta)
            elif z_max_type == 2:
                z_max_func = interpolate.interp1d(beta_table, z_max_table, kind='cubic', fill_value='extrapolate') #Interpolation of the table
                z_max = z_max_func(beta)
            #--------------------------------------------------------------------------

            #Distance from keel/water-line intersection to start of wetted chine (Eq. 9.10 of Faltinsen)
            x_s = 0.5 * b * np.tan(pi/180*beta) / ((1 + z_max) * (pi/180)*(tau + eta_5))
            alpha = np.arctan(b/(2*x_s))*180/pi #Angle between spray line and keel (projected to plan view)
            if x_s < 0:
                x_s = 0

            #Chine wetted length, Eq. 9.51 of Faltinsen 2005
            L_C = L_K - x_s
            if L_C < 0:
                L_C = 0
                x_s = L_K
                warnings.warn('Vessel operating without wetted chines (L_C = 0).', stacklevel=2)

            #Mean wetted length-to-beam ratio
            lambda_W = (L_K + L_C) / (2 * b)
            
        elif wetted_lengths_type == 2:
            #Eq. 3 of Savitsky '64
            x_s = b/pi*np.tan(pi/180*beta)/np.tan(pi/180*(tau + eta_5))
            alpha = np.arctan(b/(2*x_s))*180/pi #Angle between spray line and keel (projected to plan view)

            #z_max/Vt coefficient (E. 9.10 of Faltinsen 2005 rearranged)
            z_max = 0.5 * b * np.tan(pi/180*beta) / (x_s * (pi/180)*(tau + eta_5)) - 1
            
            #Chine wetted length
            L_C = L_K - x_s
            if L_C < 0:
                L_C = 0
                x_s = L_K
                warnings.warn('Vessel operating without wetted chines (L_C = 0).', stacklevel=2)
            
            #Mean wetted length-to-beam ratio
            lambda_W = (L_K + L_C)/(2*b)
        
        elif wetted_lengths_type == 3:
            #Eq. 12 of Savitsky '76
            w = (0.57 + beta/1000)*(np.tan(pi/180*beta)/(2*np.tan(pi/180*(tau+eta_5)))-beta/167)

            lambda_K = L_K/b

            #Eq. 14 of Savitsky '76 
            lambda_C = (lambda_K-w)-0.2*np.exp(-(lambda_K-w)/0.3)
            L_C = lambda_C*b

            x_s = L_K - L_C
            alpha = np.arctan(b/(2*x_s))*180/pi #Angle between spray line and keel (projected to plan view)

            #z_max/Vt coefficient (Eq. 9.10 of Faltinsen 2005 rearranged)
            z_max = 0.5 * b * np.tan(pi/180*beta) / (x_s * (pi/180)*(tau + eta_5)) - 1

            if lambda_C < 0: #Need to carry over L_C<0 on above formulas to not change the angles
                lambda_C = 0
                L_C = 0
                x_s = L_K
                warnings.warn('Vessel operating without wetted chines (L_C = 0). Wetted surfaces calculated using wetted_lengths_type=3 (Savitsky \'76) may be unreliable. Check alpha and z_max (z_max/Ut) values. Consider using wetted_lengths_type=1 or 2.', stacklevel=2)

            #Mean wetted length-to-beam ratio, Eq. 15 of Savitsky '76
            lambda_W = (lambda_K + lambda_C)/2+0.03
        
        if self.loa is not None:
            if L_K > self.loa:
                warnings.warn('The estimated wetted chine length ({0:.3f}) is larger than the vessel overall length ({1:.3f}).'.format(L_K, self.loa), stacklevel=2)

        #Chines-dry planing condition (Eq. 3 of Savitsky '76)
        Fn_B = U/np.sqrt(g*b) #Beam Froude number
        chines_dry = Fn_B**2 - (lambda_W - 0.16*np.tan(beta*pi/180)/np.tan(tau*pi/180))/(3*np.sin(tau*pi/180))        
        if chines_dry >= 0:
            L_C2 = 0
        else:
            L_C2 = L_C - 3*U**2*np.sin(tau*pi/180)/g #Side wetting length (Eq. 1 of Savitsky '76)

        #Transom draft
        T = L_K*np.sin((tau+eta_5)*pi/180)
        
        #Update values
        self.L_K = L_K
        self.L_C = L_C
        self.L_C2 = L_C2
        self.lambda_W = lambda_W
        self.x_s = x_s
        self.alpha = alpha
        self.z_max = z_max
        self.T = T
    
    def get_forces(self, runGeoLengths=True):
        """This function calls all the force functions to update the respective object attributes.
        
        Adds/updates the following attributes:

        - :attr:`C_Lbeta`
        - :attr:`lcp`
        - :attr:`wetted_bottom_area`
        - :attr:`bottom_fluid_speed`
        - :attr:`C_f`
        - :attr:`deltaC_f`
        - :attr:`deltaC_L`
        - :attr:`hydrodynamic_force`
        - :attr:`skin_friction`
        - :attr:`lift_change`
        - :attr:`air_resistance`
        - :attr:`flap_force`
        - :attr:`thrust_force`
        - :attr:`net_force`
        
        Args:
            runGeoLengths (boolean, optional): Calculate the wetted lengths before calculating the forces. Defaults to True.

        Methods:
            get_hydrodynamic_force(): This function follows Savitsky 1964 and Faltinsen 2005 in calculating the vessel's hydrodynamic forces and moment.
            get_skin_friction(): This function outputs the frictional force of the vessel using ITTC 1957 and the Townsin 1985 roughness allowance.
            get_lift_change(): This function estimates the lift change due to roughness wr.r.t. global coordinates.
            get_air_resistance(): This function estimates the air drag. It assumes a square shape projected area with a shape coefficient.
            get_flap_force(): This function outputs the flap forces w.r.t. global coordinates (Savitsky & Brown 1976). Horz: Positive Aft, Vert: Positive Up, Moment: Positive CCW.
            sum_forces(): This function gets the sum of forces and moments, and consequently the required net thrust. The coordinates are positive aft, positive up, and positive counterclockwise.
        """
        if runGeoLengths:
            self.get_geo_lengths() #Calculated wetted lengths in get_forces()
        
        g = self.g 
        rho_air = self.rho_air
        C_D = self.C_D
        rho = self.rho
        nu = self.nu
        AHR = self.ahr
        LD_change = self.LD_change
        W = self.weight
        epsilon = self.epsilon
        vT = self.vT
        lT = self.lT
        U = self.speed
        b = self.beam
        lcg = self.lcg
        vcg = self.vcg
        
        Lf = self.Lf
        sigma = self.sigma
        delta = self.delta
        beam = self.beam
        
        l_air = self.l_air
        h_air = self.h_air
        b_air = self.b_air
        C_shape = self.C_shape
        
        z_wl = self.z_wl
        tau = self.tau
        beta = self.beta
        eta_3 = self.eta_3
        eta_5 = self.eta_5
        
        L_K = self.L_K
        L_C = self.L_C
        lambda_W = self.lambda_W
        x_s = self.x_s
        alpha = self.alpha
        z_max = self.z_max

        roughness_penalty_type = self.roughness_penalty_type
        
        pi = np.pi
                

        #Call functions
        self._sum_forces(U, rho, b, lcg, vcg, tau, beta, g, epsilon, vT, lT, Lf, sigma, delta, 
                         C_shape, b_air, h_air, l_air, C_D, rho_air, AHR, LD_change, 
                         roughness_penalty_type, pi, eta_5, lambda_W, x_s, alpha, L_K, L_C)
    
    def _get_hydrodynamic_force(self, U, rho, b, lcg, tau, beta, g, pi, eta_5, lambda_W):
        """This function follows Savitsky 1964 and Faltinsen 2005 in calculating the vessel's hydrodynamic forces and moment.
        """
        #Beam Froude number
        Fn_B = U/np.sqrt(g*b)
        
        #Warnings
        if Fn_B < 0.6 or Fn_B > 13:
            warnings.warn('Beam Froude number = {0:.3f}, outside of range of applicability (0.60 <= U/sqrt(g*b) <= 13.00) for planing lift equation. Results are extrapolations.'.format(Fn_B), stacklevel=2)
        if lambda_W > 4:
            warnings.warn('Mean wetted length-beam ratio = {0:.3f}, outside of range of applicability (lambda <= 4) for planing lift equation. Results are extrapolations.'.format(lambda_W), stacklevel=2)
        if tau < 2 or tau > 15:
            warnings.warn('Vessel trim = {0:.3f}, outside of range of applicability (2 deg <= tau <= 15 deg) for planing lift equation. Results are extrapolations.'.format(tau), stacklevel=2)

        #0-Deadrise lift coefficient
        C_L0 = (tau + eta_5)**1.1 * (0.012 * lambda_W**0.5 + 0.0055 * lambda_W**2.5 / Fn_B**2)

        #Lift coefficient with deadrise, C_Lbeta
        C_Lbeta = C_L0 - 0.0065 * beta * C_L0**0.6

        #Vertical force (lift)
        F_z = C_Lbeta * 0.5 * rho * U**2 * b**2

        #Horizontal force
        F_x = F_z*np.tan(pi/180*(tau + eta_5))

        #Lift's Normal force w.r.t. keel
        F_N = F_z / np.cos(pi/180*(tau + eta_5))

        #Longitudinal position of the center of pressure, l_p (Eq. 4.41, Doctors 1985)
        l_p = lambda_W * b * (0.75 - 1 / (5.21 * (Fn_B / lambda_W)**2 + 2.39)) #Limits for this is (0.60 < Fn_B < 13.0, lambda < 4.0)

        #Moment about CG (Axis consistent with Fig. 9.24 of Faltinsen (P. 366)
        M_cg = - F_N * (lcg - l_p)
        
        #Update values
        self.C_Lbeta = C_Lbeta
        self.lcp = l_p
        self.hydrodynamic_force = np.array([F_x, F_z, M_cg])
        
    def _get_skin_friction(self, U, rho, b, vcg, tau, beta, g, nu, pi, eta_5, lambda_W, x_s, alpha, L_K, L_C, AHR, roughness_penalty_type):
        """This function outputs the frictional force of the vessel using ITTC 1957 and the Townsin 1985 roughness allowance.
        """
        #Surface area of the non-wetted-chine region
        S1 = x_s**2 * np.tan(alpha*pi/180) / np.cos(pi/180*beta)

        #Surface area of the wetted-chine region
        S2 = b * L_C / np.cos(pi/180*beta) 

        #Total surface area
        S = S1 + S2 
        if S == 0:
            V_m = 0
            C_f = 0
            deltaC_f = 0 
            F_x = 0
            F_z = 0
            M_cg = 0
        else:
            #Beam Froude number
            Fn_B = U/np.sqrt(g*b)
            
            #Warnings
            if Fn_B < 1.0 or Fn_B > 13:
                warnings.warn('Beam Froude number = {0:.3f}, outside of range of applicability (1.0 <= U/sqrt(g*b) <= 13.00) for average bottom velocity estimate. Results are extrapolations.'.format(Fn_B), stacklevel=2)

            #Mean bottom fluid velocity, Savitsky 1964 - derived to include deadrise effect
            V_m = U * np.sqrt(1 - (0.012 * tau**1.1 * np.sqrt(lambda_W) - 0.0065 * beta * (0.012 * np.sqrt(lambda_W) * tau**1.1)**0.6) / (lambda_W * np.cos(tau * pi/180)))

            #Reynolds number (with bottom fluid velocity)
            Rn = V_m * lambda_W * b / nu

            #'Friction coefficient' ITTC 1957
            C_f = 0.075/(np.log10(Rn) - 2)**2

            #Additional 'friction coefficient' due to skin friction
            if AHR > 0:
                if roughness_penalty_type == 1: #Mosaad 1986 roughness allowance
                    deltaC_f = (6*Rn**0.093*((AHR/(lambda_W*b))**(1/3) - 5.8*Rn**(-1/3)))/10**3
                elif roughness_penalty_type == 2: #Townsin 1984 roughness allowance
                    deltaC_f = (44*((AHR/(lambda_W*b))**(1/3) - 10*Rn**(-1/3)) + 0.125)/10**3
            else:
                deltaC_f = 0

            #Frictional force
            R_f = 0.5 * rho * (C_f + deltaC_f) * S * U**2

            #Geometric vertical distance from keel
            l_f = (b / 4 * np.tan(pi/180*beta) * S2 + b / 6 * np.tan(pi/180*beta) * S1) / (S1 + S2)

            #Horizontal force
            F_x = R_f * np.cos(pi/180*(tau + eta_5))

            #Vertical force
            F_z = - R_f * np.sin(pi/180*(tau + eta_5))

            #Moment about CG (Axis consistent with Fig. 9.24 of Faltinsen (P. 366))
            M_cg = R_f * (l_f - vcg)
            
        #Update values
        self.wetted_bottom_area = S
        self.bottom_fluid_speed = V_m
        self.C_f = C_f
        self.deltaC_f = deltaC_f
        self.skin_friction = np.array([F_x, F_z, M_cg])
    
    def _get_lift_change(self, U, rho, b, lcg, tau, pi, eta_5, LD_change):
        """This function estimates the lift change due to roughness wr.r.t. global coordinates.
        """
        if LD_change is None:
            self.lift_change = np.array([0, 0, 0])
            self.deltaC_L = 0
            return

        S = self.wetted_bottom_area
        if S == 0: 
            deltaR_f = 0
        else:
            #Frictional force due to roughness only
            deltaR_f = 0.5 * rho * self.deltaC_f * S * U**2
 
        #Change of hydrodynamic normal force based on ITTC '78 report on propeller tests (P. 274)
        deltaF_N = deltaR_f * (LD_change*np.cos(pi/180*(tau + eta_5)) + np.sin(pi/180*(tau + eta_5))) / (LD_change*np.sin(pi/180*(tau + eta_5)) - np.cos(pi/180*(tau + eta_5)))

        #Horizontal force
        F_x = - deltaF_N * np.sin(pi/180*(tau + eta_5))

        #Vertical force (lift)
        F_z = - deltaF_N * np.cos(pi/180*(tau + eta_5))

        #Moment about CG (Axis consistent with Fig. 9.24 of Faltinsen (P. 366)
        M_cg = deltaF_N * (lcg - self.lcp)          

        #Change in hydrodynamic lift coefficient due to roughness (Note: Does not include lift change due to roughness)
        deltaC_L = F_z/(0.5 * rho * U**2 * b**2) 

        #Update values
        self.deltaC_L = deltaC_L
        self.lift_change = np.array([F_x, F_z, M_cg])
            
    def _get_air_resistance(self, U, z_wl, l_air, L_K, h_air, b_air, C_shape, tau, pi, eta_5, rho_air, C_D):
        """This function estimates the air drag. It assumes a square shape projected area with a shape coefficient.
        """
        if C_shape == 0 or b_air == 0:
            self.air_resistance = np.array([0, 0, 0])
            return

        #Vertical distance from running calm water line to keel at l_air
        a_dist = np.sin(pi/180*(tau + eta_5))*(l_air-L_K)
        
        #Vertical distance from keel at l_air to horizontal line level with h_air
        b_dist = np.cos(pi/180*(tau + eta_5))*h_air
        
        #Vertical distance from CG to center of square (moment arm, positive is CG above)
        momArm = z_wl - (a_dist + b_dist)/2
        
        #Square projected area
        Area = (a_dist+b_dist)*b_air*C_shape
        if Area < 0:
            Area = 0
            
        #Horizontal force (Positive aft)
        F_x = 0.5*rho_air*C_D*Area*U**2
        
        #Vertical force (Positive up) 
        F_z = 0
        
        #Moment (positve CCW)
        M_cg = -F_x*momArm

        #Update values
        self.air_resistance = np.array([F_x, F_z, M_cg])
    
    def _get_flap_force(self, U, b, Lf, sigma, delta, tau, pi, eta_5, rho, lcg, g):
        """This function outputs the flap forces w.r.t. global coordinates (Savitsky & Brown 1976). Horz: Positive Aft, Vert: Positive Up, Moment: Positive CCW.
        """
        if Lf == 0:
            self.flap_force = np.array([0, 0, 0])
            return
        
        #Warnings
        if Lf > 0.10*(self.L_K + self.L_C)/2 or Lf < 0:
            warnings.warn('Flap chord = {0:.3f} outside of bounds (0-10% of mean wetted length) for flap forces estimates with Savitsky & Brown 1976'.format(Lf), stacklevel=2)
        if delta < 0 or delta > 15:
            warnings.warn('Flap deflection angle = {0:.3f} out of bounds (0-15 deg) for flap forces estimates with Savitsky & Brown 1976'.format(delta), stacklevel=2)
        Fn_B = U/np.sqrt(g*b)
        if Fn_B < 2 or Fn_B > 7:
            warnings.warn('Beam-based Froude number Fn_B = {0:.3f} out of bounds (2-7) for flap forces estimates with Savitsky & Brown 1976'.format(Fn_B), stacklevel=2)
        
        F_z = 0.046*(Lf*3.28084)*delta*sigma*(b*3.28084)*(rho/515.379)/2*(U*3.28084)**2*4.44822

        F_x = 0.0052*F_z*(tau+eta_5+delta)

        l_flap = 0.6*b+Lf*(1-sigma)

        M_cg = -F_z*(lcg-l_flap)
        
        #Update values
        self.flap_force = np.array([F_x, F_z, M_cg])
    
    def _sum_forces(self, U, rho, b, lcg, vcg, tau, beta, g, epsilon, vT, lT, Lf, sigma, delta, 
                    C_shape, b_air, h_air, l_air, C_D, rho_air, AHR, LD_change, 
                    roughness_penalty_type, pi, eta_5, lambda_W, x_s, alpha, L_K, L_C):
        """This function gets the sum of forces and moments, and consequently the required net thrust. The coordinates are positive aft, positive up, and positive counterclockwise.
        """
        #Call all force functions-------
        self._get_hydrodynamic_force(U, rho, b, lcg, tau, beta, g, pi, eta_5, lambda_W)
        self._get_skin_friction(U, rho, b, vcg, tau, beta, g, self.nu, pi, eta_5, lambda_W, x_s, alpha, L_K, L_C, AHR, roughness_penalty_type)
        self._get_lift_change(U, rho, b, lcg, tau, pi, eta_5, LD_change)
        self._get_air_resistance(U, self.z_wl, l_air, L_K, h_air, b_air, C_shape, tau, pi, eta_5, rho_air, C_D)
        self._get_flap_force(U, b, Lf, sigma, delta, tau, pi, eta_5, rho, lcg, g)
        #-------------------------------
        
        forcesMatrix = np.column_stack((self.gravity_force, self.hydrodynamic_force, self.skin_friction, self.lift_change, self.air_resistance, self.flap_force)) #Forces and moments
        F_sum = np.sum(forcesMatrix, axis=1) #F[0] is x-dir, F[1] is z-dir, and F[2] is moment

        #Required thrust and resultant forces
        thrust = F_sum[0]/np.cos(pi/180*(epsilon+tau+eta_5)); #Magnitude
        thrust_z = thrust*np.sin(pi/180*(epsilon+tau+eta_5)); #Vertical
        thrust_cg = thrust*np.cos(pi/180*epsilon)*(vcg - vT) - thrust*np.sin(pi/180*epsilon)*(lcg - lT); #Moment about cg
        
        #Update resultant thurst values
        self.thrust_force = np.array([-F_sum[0], thrust_z, thrust_cg])
        
        #Include resultant thrust forces in sum
        F_sum[1] = F_sum[1]+thrust_z
        F_sum[2] = F_sum[2]+thrust_cg
        
        #Update values
        self.net_force = F_sum
        
    def get_steady_trim(self, x0=[0, 3], tauLims=[0.5, 35], tolF=10**-6, maxiter=50):
        """This function finds and sets the equilibrium point when the vessel is steadily running in calm water.
        
        Updates the following attributes:

        - :attr:`z_wl`
        - :attr:`tau`

        Args:
            x0 (list of float): Initial guess for equilibirum point [z_wl (m), tau (deg)]. Defaults to [0, 3].
            tauLims (list of float): Limits for equilibrium trim search. Defaults to [0.5, 35].
            tolF (float): Tolerance for convergence to zero. Defaults to 10**-6.
            maxiter (float): Maximum iterations. Defaults to 50.
        """
        def _boatForces(x):
            self.z_wl = x[0]/10 #the division is for normalization of the variables
            self.tau = x[1]
            self.get_forces()
            return self.net_force[1:3]

        def _boatForcesPrime(x):
            return ndmath.complexGrad(_boatForces, x)

        def _L_K_constraint(x):
            # self.z_wl = x[0]/10
            # self.tau = x[1]
            # self.get_geo_lengths() #No need to call, because ndmath's nDimNewton allways calls the obj function before calling this "constraint"
            return [-self.L_K]
        
        xlims = np.array([[-np.inf, np.inf], tauLims])
        warnings.filterwarnings("ignore", category=UserWarning)
        [self.z_wl, self.tau] = ndmath.nDimNewton(_boatForces, x0, _boatForcesPrime, tolF, maxiter, xlims, hehcon=_L_K_constraint)/[10, 1]
        warnings.filterwarnings("default", category=UserWarning)
        
    def _calculate_mass_matrix(self, W, rho, b, lcg, tau, beta, g, r_g, pi, eta_5, L_K, L_C, lambda_W, x_s, z_max):
        """Calculate mass matrix coefficients following Sec. 9.4.1 of Faltinsen 2005.
        
        Args:
            W (float): Weight (N)
            rho (float): Water density (kg/m^3)
            b (float): Beam (m)
            lcg (float): Longitudinal center of gravity (m)
            tau (float): Trim angle (deg)
            beta (float): Deadrise (deg)
            g (float): Gravitational acceleration (m/s^2)
            r_g (float): Radius of gyration (m)
            pi (float): Pi constant
            eta_5 (float): Additional trim (deg)
            L_K (float): Keel wetted length (m)
            L_C (float): Chine wetted length (m)
            lambda_W (float): Mean wetted length-beam ratio
            x_s (float): Distance to start of wetted chine (m)
            z_max (float): Maximum pressure coordinate coefficient
        """
        #Distance of CG from keel-WL intersection
        x_G = L_K - lcg

        #K constant (Eq. 9.63 of Faltinsen 2005)
        K = (pi / np.sin(pi/180*beta) * gamma(1.5 - beta/180) / (gamma(1 - beta/180)**2 * gamma(0.5 + beta/180)) - 1) / np.tan(pi/180*beta)

        kappa = (1 + z_max) * (pi/180)*(tau + eta_5) #User defined constant

        #Based on Faltinsen's
        A1_33 = rho * kappa**2 * K * x_s**3 / 3
        A1_35 = A1_33 * (x_G - x_s * 3/4)
        A1_53 = A1_35
        A1_55 = A1_33 * (x_G**2 - 3/2 * x_G * x_s + 3/5 * x_s**2)

        #Contribution from wet-chine region
        if L_C > 0:
            C_1 = 2 * np.tan(pi/180*beta)**2 / pi * K

            A2_33 = (rho * b**3) * C_1 * pi / 8 * L_C / b
            A2_35 = (rho * b**4) * (- C_1 * pi / 16 * ((L_K / b)**2 - (x_s / b)**2) + x_G / b * A2_33 / (rho * b**3))
            A2_53 = A2_35
            A2_55 = (rho * b**5) * (C_1 * pi / 24 * ((L_K / b)**3 - (x_s / b)**3) - C_1 / 8 * pi * (x_G / b) * ((L_K / b)**2 - (x_s / b)**2) + (x_G / b)**2 * A2_33 / (rho * b**3))
        else:
            A2_33 = 0
            A2_35 = 0
            A2_53 = 0
            A2_55 = 0

        #Total added mass & update values
        A_33 = A1_33 + A2_33 + W/g # kg, A_33
        A_35 = A1_35 + A2_35 # kg*m/rad, A_35
        A_53 = A1_53 + A2_53 # kg*m, A_53
        A_55 = A1_55 + A2_55 + W/g*r_g**2 # kg*m^2/rad, A_55
        self.mass_matrix = np.array([[A_33, A_35], [A_53, A_55]])
    
    def _calculate_damping_matrix(self, U, rho, b, lcg, tau, beta, g, pi, eta_5, L_K, L_C, lambda_W, x_s, z_max):
        """Calculate damping matrix coefficients following Sec. 9.4.1 of Faltinsen 2005.
        
        Args:
            U (float): Speed (m/s)
            rho (float): Water density (kg/m^3)
            b (float): Beam (m)
            lcg (float): Longitudinal center of gravity (m)
            tau (float): Trim angle (deg)
            beta (float): Deadrise (deg)
            g (float): Gravitational acceleration (m/s^2)
            pi (float): Pi constant
            eta_5 (float): Additional trim (deg)
            L_K (float): Keel wetted length (m)
            L_C (float): Chine wetted length (m)
            lambda_W (float): Mean wetted length-beam ratio
            x_s (float): Distance to start of wetted chine (m)
            z_max (float): Maximum pressure coordinate coefficient
        """
        #Heave-heave added mass (need to substract W/g since it was added)
        A_33 = self.mass_matrix[0,0] - self.weight/g

        if L_C > 0:
            d = 0.5 * b * np.tan(pi/180*beta)
        else:
            d = (1 + z_max) * (pi/180)*(tau + eta_5) * L_K

        #K constant (Eq. 9.63 of Faltinsen 2005, P. 369)
        K = (pi / np.sin(pi/180*beta) * gamma(1.5 - beta/180) / (gamma(1 - beta/180)**2 * gamma(0.5 + beta/180)) - 1) / np.tan(pi/180*beta)

        #2D Added mass coefficient in heave
        a_33 = rho * d**2 * K

        #Infinite Fn lift coefficient
        C_L0 = (tau + eta_5)**1.1 * 0.012 * lambda_W**0.5

        #Derivative w.r.t. tau (rad) of inf. Fn C_L0
        dC_L0 = (180 / pi)**1.1 * 0.0132 * (pi/180*(tau + eta_5))**0.1 * lambda_W**0.5

        #Derivative w.r.t. tau (rad) of inf. Fn C_Lbeta
        dC_Lbeta = dC_L0 * (1 - 0.0039 * beta * C_L0**-0.4)

        #Damping coefficients & update values
        B_33 = rho / 2 * U * b**2 * dC_Lbeta # kg/s, B_33, Savitsky based
        B_35 = - U * (A_33 + lcg * a_33) # kg*m/(s*rad), B_35, Infinite frequency based
        B_53 = B_33 * (0.75 * lambda_W * b - lcg) # kg*m/s, B_53, Savitsky based
        B_55 = U * lcg**2 * a_33 # kg*m**2/(s*rad), B_55, Infinite frequency based
        self.damping_matrix = np.array([[B_33, B_35], [B_53, B_55]])
    
    def _calculate_restoring_matrix(self, pi, diffType=1, step=10**-6.6):
        """Calculate restoring matrix coefficients following the approach in Sec. 9.4.1 of Faltinsen 2005.
        
        Args:
            pi (float): Pi constant
            diffType (int, optional): 1 (recommended) = Complex step method, 2 = Foward step difference. Defaults to 1.
            step (float, optional): Step size if using diffType == 2. Defaults to 10**-6.
        """
        def _func(eta):
            self.eta_3 = eta[0] 
            self.eta_5 = eta[1]
            self.get_forces() #This needs to run get_geo_lengths() to work
            return self.net_force[1:3]
        
        temp_eta_3 = self.eta_3
        temp_eta_5 = self.eta_5
        
        warnings.filterwarnings("ignore", category=UserWarning)
        if diffType == 1:
            C_full = -ndmath.complexGrad(_func, [temp_eta_3, temp_eta_5])
        elif diffType == 2:
            C_full = -ndmath.finiteGrad(_func, [temp_eta_3, temp_eta_5], step)

        #Reset values
        self.eta_3 = temp_eta_3
        self.eta_5 = temp_eta_5
        self.get_forces()
        warnings.filterwarnings("default", category=UserWarning)
        
        #Conversion deg to rad (degree in denominator)
        C_full[0,1] = C_full[0,1] / (pi/180) # N/rad, C_35
        C_full[1,1] = C_full[1,1] / (pi/180) # N*m/rad, C_55
        
        #Update values
        self.restoring_matrix = C_full
    
    def get_eom_matrices(self, runGeoLengths=True):
        """This function returns the mass, damping, and stiffness matrices following Faltinsen 2005.

        Adds/updates the following parameters:
            
        - :attr:`mass_matrix`
        - :attr:`damping_matrix`
        - :attr:`restoring_matrix`

        Args:
            runGeoLengths (boolean, optional): Calculate the wetted lengths before calculating the EOM matrices. Defaults to True.
        """
        if runGeoLengths:
            self.get_geo_lengths() #Calculated wetted lengths in get_eom_matrices()
        
        W = self.weight
        U = self.speed
        rho = self.rho
        b = self.beam
        lcg = self.lcg
        tau = self.tau
        beta = self.beta
        g = self.g
        r_g = self.r_g
        
        eta_5 = self.eta_5
        
        L_K = self.L_K
        L_C = self.L_C
        lambda_W = self.lambda_W
        x_s = self.x_s
        z_max = self.z_max
        
        pi = np.pi
        
        #Call functions
        self._calculate_mass_matrix(W, rho, b, lcg, tau, beta, g, r_g, pi, eta_5, L_K, L_C, lambda_W, x_s, z_max)
        self._calculate_damping_matrix(U, rho, b, lcg, tau, beta, g, pi, eta_5, L_K, L_C, lambda_W, x_s, z_max)
        self._calculate_restoring_matrix(pi)
    
    def check_porpoising(self, stepEstimateType=1):
        """This function checks for porpoising.

        Adds/updates the following parameters:
        
        - :attr:`porpoising` (list):
        
        Args:
            stepEstimateType (int, optional): Pitch step response settling time estimate type, 1 = -3/np.real(eigVals[0])], 2 = Time-domain simulation estimate. Defaults to 1.        
        """
        #Eigenvalue analysis
        try:
            self.mass_matrix
        except AttributeError:
            warnings.warn('No Equation Of Motion (EOM) matrices found. Running get_eom_matrices().', stacklevel=2)
            self.get_eom_matrices()
            
        M = self.mass_matrix
        C = self.damping_matrix
        K = self.restoring_matrix
        
        nDim = len(M)
        A_ss = np.concatenate((np.concatenate((np.zeros((nDim,nDim)), np.identity(nDim)), axis=1), np.concatenate((-np.linalg.solve(M,K), -np.linalg.solve(M,C)), axis=1))) #State space reprecentation 
        
        eigVals = np.linalg.eigvals(A_ss)

        eig_porpoise = any(eigVal >= 0 for eigVal in eigVals)
        
        if stepEstimateType == 1:
            settling_time = -3/np.real(eigVals[0])
        elif stepEstimateType == 2: 
            B_ss = np.array([[1],[0],[0],[0]]) #Pitch only
            C_ss = np.array([[1,0,0,0]]) #Pitch only
            D_ss = np.array([[0]])
            
            system = (A_ss,B_ss,C_ss,D_ss)
            t, y = signal.step(system)
            settling_time = (t[next(len(y)-i for i in range(2,len(y)-1) if abs(y[-i]/y[-1])>1.02)]-t[0])
        
        #Savitsky '64 chart method
        C_L = self.weight/(1/2*self.rho*self.speed**2*self.beam**2)
        x = np.sqrt(C_L/2)

        #Warnings
        if x > 0.3 or x < 0.13:
            warnings.warn('Lift Coefficient = {0:.3f} outside of bounds (0.0338-0.18) for porpoising estimates with Savitsky 1964. Results are extrapolations.'.format(C_L), stacklevel=2)
        if self.beta > 20:
            warnings.warn('Deadrise = {0:.3f} outside of bounds (0-20 deg) for porpoising estimates with Savitsky 1964. Results are extrapolations.'.format(self.beta), stacklevel=2)

        tau_crit_0 = -376.37*x**3 + 329.74*x**2 - 38.485*x + 1.3415
        tau_crit_10 = -356.05*x**3 + 314.36*x**2 - 41.674*x + 3.5786
        tau_crit_20 = -254.51*x**3 + 239.65*x**2 - 23.936*x + 3.0195

        tau_crit_func = interpolate.interp1d([0, 10, 20], [tau_crit_0, tau_crit_10, tau_crit_20], kind='quadratic', fill_value='extrapolate')
        tau_crit = tau_crit_func(self.beta)

        if self.tau > tau_crit:
            chart_porpoise = True
        else:
            chart_porpoise = False
        
        #Update values
        self.porpoising = [[eig_porpoise, settling_time], [chart_porpoise, float(tau_crit)]]
        
    def get_seaway_behavior(self):
        """This function calculates the seaway behavior as stated in Savitsky & Brown '76.
        
        Adds/updates the following parameters:
        
        - :attr:`avg_impact_acc`
        - :attr:`R_AW`
        """
        if self.H_sig is None:
            self.H_sig = self.beam*0.5 #Arbitrary wave height if no user-defined wave height
            warnings.warn('Significant wave height has not been specified. Using beam*0.5 = {0:.3f} m.'.format(self.H_sig), stacklevel=2)
        if self.loa is None:
            self.loa = self.beam*3
            warnings.warn('Vessel overall length has not been specified. Using beam*3 = {0:.3f} m.'.format(self.loa), stacklevel=2)
        H_sig = self.H_sig
        
        W = self.weight
        beta = self.beta
        tau = self.tau
        
        pi = np.pi
        
        Delta_LT = W/9964 #Displacement in long tons
        Delta = Delta_LT*2240 #Displacement in lbf
        L = self.loa*3.281 #Length in ft
        b = self.beam*3.281 #Beam in ft
        Vk = self.speed*1.944 #Speed in knots
        Vk_L = Vk/np.sqrt(L) #Vk/sqrt(L)
        H_sig = H_sig*3.281 #Significant wave height in ft
        
        w = self.rho*self.g/(4.448*35.315) #Specific weight in lbf/ft^3
        
        C_Delta = Delta/(w*b**3) #Static beam-loading coefficient

        if self.seaway_drag_type == 1: #Savitsky '76
            #Check that variables are inside range of applicability (P. 395 of Savitsky & Brown '76)
            P1 = Delta_LT/(0.01*L)**3
            P2 = L/b
            P5 = H_sig/b
            P6 = Vk_L
            if P1 < 100 or P1 > 250:
                warnings.warn('Vessel displacement coefficient = {0:.3f}, outside of range of applicability (100 <= Delta_LT/(0.01*L)^3 <= 250, with units LT/ft^3). Results are extrapolations.'.format(P1), stacklevel=2)
            if P2 < 3 or P2 > 5:
                warnings.warn('Vessel overall length/beam = {0:.3f}, outside of range of applicability (3 <= L/b <= 5). Results are extrapolations.'.format(P2), stacklevel=2)
            if tau < 3 or tau > 7:
                warnings.warn('Vessel trim = {0:.3f}, outside of range of applicability (3 deg <= tau <= 7 deg). Results are extrapolations.'.format(tau), stacklevel=2)
            if beta < 10 or beta > 30:
                warnings.warn('Vessel deadrise = {0:.3f}, outside of range of applicability (10 deg <= beta <= 30 deg). Results are extrapolations.'.format(beta), stacklevel=2)
            if P5 < 0.2 or P5 > 0.7:
                warnings.warn('Significant wave height / beam = {0:.3f}, outside of range of applicability (0.2 <= H_sig/b <= 0.7). Results are extrapolations.'.format(P5), stacklevel=2)
            if P6 < 2 or P6 > 6:
                warnings.warn('Speed coefficient = {0:.3f}, outside of range of applicability (2 <= Vk/sqrt(L) <= 6, with units knots/ft^0.5). Results are extrapolations.'.format(P6), stacklevel=2)
                
            R_AW_2 = (w*b**3)*66*10**-6*(H_sig/b+0.5)*(L/b)**3/C_Delta+0.0043*(tau-4) #Added resistance at Vk/sqrt(L) = 2
            R_AW_4 = (Delta)*(0.3*H_sig/b)/(1+2*H_sig/b)*(1.76-tau/6-2*np.tan(beta*pi/180)**3) #Vk/sqrt(L) = 4
            R_AW_6 = (w*b**3)*(0.158*H_sig/b)/(1+(H_sig/b)*(0.12*beta-21*C_Delta*(5.6-L/b)+7.5*(6-L/b))) #Vk/sqrt(L) = 6
            R_AWs = np.array([R_AW_2, R_AW_4, R_AW_6])
            
            R_AWs_interp = interpolate.interp1d([2,4,6], R_AWs, kind='quadratic', fill_value='extrapolate')
            R_AW = R_AWs_interp([Vk_L])[0]

        elif self.seaway_drag_type == 2: #Fridsma '71 design charts
            #Check that variables are inside range of applicability (P. R-1495 of Fridsma '71)
            if C_Delta < 0.3 or C_Delta > 0.9:
                warnings.warn('C_Delta = {0:.3f}, outside of range of applicability (0.3 <= C_Delta <= 0.9). Results are extrapolations'.format(C_Delta), stacklevel=2)
            if L/b < 3 or L/b > 6:
                warnings.warn('L/b = {0:.3f}, outside of range of applicability (3 <= L/b <= 6). Results are extrapolations'.format(L/b), stacklevel=2)
            if C_Delta/(L/b) < 0.06 or C_Delta/(L/b) > 0.18:
                warnings.warn('C_Delta/(L/b) = {0:.3f}, outside of range of applicability (0.06 <= C_Delta/(L/b) <= 0.18). Results are extrapolations'.format(C_Delta/(L/b)), stacklevel=2)
            if tau < 3 or tau > 7:
                warnings.warn('tau = {0:.3f}, outside of range of applicability (3 <= tau <= 7). Results are extrapolations'.format(tau), stacklevel=2)
            if beta < 10 or beta > 30:
                warnings.warn('beta = {0:.3f}, outside of range of applicability (10 <= beta <= 30). Results are extrapolations'.format(beta), stacklevel=2)
            if H_sig/b > 0.8:
                warnings.warn('H_sig/b = {0:.3f}, outside of range of applicability (H_sig/b <= 0.8). Results are extrapolations'.format(H_sig/b), stacklevel=2)
            if Vk_L > 6:
                warnings.warn('Vk_L = {0:.3f}, outside of range of applicability (Vk_L <= 6). Results are extrapolations'.format(Vk_L), stacklevel=2)

            # Use embedded table data (no file I/O required)
            arr_Raw2 = RAW_02_DATA
            arr_Raw4 = RAW_04_DATA
            arr_Raw6 = RAW_06_DATA

            arr_V2 = V_02_DATA
            arr_V4 = V_04_DATA

            arr_Raw_V2 = RAW_V_02_DATA
            arr_Raw_V4 = RAW_V_04_DATA
            arr_Raw_V6 = RAW_V_06_DATA
            
            #Create interpolation functions
            interp1Type = 'linear'
            interp2Type = 'linear'

            Raw2m_interp = interpolate.interp2d(arr_Raw2[:, 1], arr_Raw2[:, 0], arr_Raw2[:, 2], kind=interp2Type)
            Raw4m_interp = interpolate.interp2d(arr_Raw4[:, 1], arr_Raw4[:, 0], arr_Raw4[:, 2], kind=interp2Type)
            Raw6m_interp = interpolate.interp2d(arr_Raw6[:, 1], arr_Raw6[:, 0], arr_Raw6[:, 2], kind=interp2Type)

            V2m_interp = interpolate.interp2d(arr_V2[:, 1], arr_V2[:, 0], arr_V2[:, 2], kind=interp2Type)
            V4m_interp = interpolate.interp2d(arr_V4[:, 1], arr_V4[:, 0], arr_V4[:, 2], kind=interp2Type)
            V6m_interp = V4m_interp

            RawRaw2m_interp = interpolate.interp1d(arr_Raw_V2[:, 0], arr_Raw_V2[:, 1], kind=interp1Type, fill_value='extrapolate')
            RawRaw4m_interp = interpolate.interp1d(arr_Raw_V4[:, 0], arr_Raw_V4[:, 1], kind=interp1Type, fill_value='extrapolate')
            RawRaw6m_interp = interpolate.interp1d(arr_Raw_V6[:, 0], arr_Raw_V6[:, 1], kind=interp1Type, fill_value='extrapolate')

            #Get values following procedure shown in Fridsma 1971 paper
            VLm = [V2m_interp(beta, tau)[0], V4m_interp(beta, tau)[0], V6m_interp(beta, tau)[0]]
            Rwbm = [Raw2m_interp(beta, tau)[0], Raw4m_interp(beta, tau)[0], Raw6m_interp(beta, tau)[0]]
            VVm = Vk_L/VLm
            RRm = [RawRaw2m_interp(VVm[0]), RawRaw4m_interp(VVm[1]), RawRaw6m_interp(VVm[2])]
            Rwb = np.multiply(RRm, Rwbm)

            E1 = lambda H_sig: 1 + ((L/b)**2/25 - 1)/(1 + 0.895*(H_sig/b - 0.6)) #V/sqrt(L) = 2
            E2 = lambda H_sig: 1 + 10*H_sig/b*(C_Delta/(L/b) - 0.12) #V/sqrt(L) = 4
            E3 = lambda H_sig: 1 + 2*H_sig/b*(0.9*(C_Delta-0.6)-0.7*(C_Delta-0.6)**2) #V/sqrt(L) = 6
            E_interp = lambda H_sig: interpolate.interp1d([2, 4, 6], [E1(H_sig), E2(H_sig), E3(H_sig)], kind=interp1Type, fill_value='extrapolate')
            E = [E_interp(0.2*b)(Vk_L), E_interp(0.4*b)(Vk_L), E_interp(0.6*b)(Vk_L)]

            Rwb_final = np.multiply(Rwb,E)
            Rwb_final_interp = interpolate.interp1d([0.2, 0.4, 0.6], Rwb_final, kind=interp1Type, fill_value='extrapolate')

            R_AW = Rwb_final_interp(H_sig/b)*w*b**3

            warnings.warn('Average impact acceleration based on the Fridsma charts is currently not implemented. Using Savitsky & Brown approximation.', stacklevel=2)

        n_cg = 0.0104*(H_sig/b+0.084)*tau/4*(5/3-beta/30)*(Vk_L)**2*L/b/C_Delta #g, at CG
        n_bow = n_cg*(1+3.8*(L/b-2.25)/(Vk_L)) #g, at bow
        avg_impact_acc = np.array([n_cg, n_bow])
        
        #Update values
        self.avg_impact_acc = avg_impact_acc
        self.R_AW = R_AW*4.448 #lbf to N conversion


# =============================================================================
# VIRTUAL PRISMATIC HULL (VPH) METHOD - Ribeiro 2002 Thesis
# =============================================================================
# Implements the "Método dos Cascos Prismáticos Virtuais" from Chapter 4 of:
# RIBEIRO, H.J.C. "Equilíbrio Dinâmico de Cascos Planadores", COPPE/UFRJ, 2002
# 
# This method extends Savitsky's approach to handle arbitrary hull forms with
# variable deadrise angle β(x) and chine line distributions using strip theory.
# =============================================================================


class VirtualPrismaticHull():
    """Non-prismatic planing hull using Virtual Prismatic Hull (VPH) method
    
    This class implements the empirical method from Ribeiro (2002) for computing
    dynamic equilibrium of planing hulls with arbitrary geometry (variable deadrise,
    chine distributions, non-prismatic forms). It uses the "Virtual Prismatic Hull"
    concept where each section is mapped to an equivalent prismatic hull, and a
    sectional lift distribution is solved via a 4x4 linear system constrained by
    global force/moment balance.
    
    Attributes:
        speed (float): Speed (m/s). Input parameter.
        weight (float): Weight (N). Input parameter.
        loa (float): Length overall (m). Input parameter.
        beam (float): Maximum beam (m). Input parameter.
        lcg (float): Longitudinal center of gravity from stern (m). Input parameter.
        vcg (float): Vertical center of gravity from keel (m). Input parameter.
        beta_dist (list or callable): Deadrise angle distribution β(x) in degrees.
            Can be a list of [x, beta] pairs or a function beta(x). Input parameter.
        chine_dist (list or callable): Chine half-beam distribution b_Q(x) in meters.
            Can be a list of [x, b_Q] pairs or a function b_Q(x). Input parameter.
        cl_dist (list or callable): Centerline height distribution CL(x) in meters.
            Can be a list of [x, CL] pairs or a function CL(x). Input parameter.
        transom_angle (float): Transom angle δ (deg). Defaults to 0. Input parameter.
        epsilon (float): Shaft inclination angle ε (deg), positive down. Defaults to 0.
        ahr (float): Average hull roughness (m). Defaults to 150e-6.
        rho (float): Water density (kg/m³). Defaults to 1025.87.
        nu (float): Water kinematic viscosity (m²/s). Defaults to 1.19e-6.
        g (float): Gravitational acceleration (m/s²). Defaults to 9.8066.
        n_sections (int): Number of sections for discretization. Defaults to 100.
        z_wl (float): Static draft at CG (m). Updated by get_steady_trim().
        tau (float): Dynamic trim angle (deg). Updated by get_steady_trim().
        L_K (float): Keel wetted length (m). Updated by get_geo_lengths().
        L_C (float): Mean wetted length (m). Updated by get_geo_lengths().
        lambda_W (float): Wetted length-to-beam ratio. Updated by get_geo_lengths().
        lcp (float): Longitudinal center of pressure from stern (m). Updated by get_forces().
        total_lift (float): Total hydrodynamic + hydrostatic lift (N). Updated by get_forces().
        skin_friction ((3,) ndarray): Skin friction forces [Fx, Fz, M]. Updated by get_forces().
        spray_resistance (float): Spray resistance component (N). Updated by get_forces().
        pressure_resistance (float): Pressure resistance component (N). Updated by get_forces().
        total_resistance (float): Total resistance (N). Updated by get_forces().
        validity_flag (bool): True if within applicability limits. Updated by check_validity().
    """
    
    def __init__(self, speed, weight, loa, beam, lcg, vcg, beta_dist, chine_dist=None, 
                 cl_dist=None, transom_angle=0, epsilon=0, ahr=150e-6, 
                 rho=1025.87, nu=1.19e-6, g=9.8066, n_sections=100):
        """Initialize VirtualPrismaticHull
        
        Args:
            speed (float): Speed (m/s).
            weight (float): Weight (N).
            loa (float): Length overall (m).
            beam (float): Maximum beam (m).
            lcg (float): LCG from stern (m).
            vcg (float): VCG from keel (m).
            beta_dist (list/callable): Deadrise distribution β(x) [deg].
            chine_dist (list/callable, optional): Chine half-beam b_Q(x) [m]. 
                If None, assumes constant beam = beam/2.
            cl_dist (list/callable, optional): Centerline height CL(x) [m].
                If None, assumes flat bottom (CL=0).
            transom_angle (float, optional): Transom angle δ [deg]. Defaults to 0.
            epsilon (float, optional): Shaft angle ε [deg], positive down. Defaults to 0.
            ahr (float, optional): Average hull roughness [m]. Defaults to 150e-6.
            rho (float, optional): Water density [kg/m³]. Defaults to 1025.87.
            nu (float, optional): Kinematic viscosity [m²/s]. Defaults to 1.19e-6.
            g (float, optional): Gravity [m/s²]. Defaults to 9.8066.
            n_sections (int, optional): Number of sections. Defaults to 100.
        """
        self.speed = speed
        self.weight = weight
        self.loa = loa
        self.beam = beam
        self.lcg = lcg
        self.vcg = vcg
        self.beta_dist = beta_dist
        self.chine_dist = chine_dist if chine_dist is not None else lambda x: beam/2
        self.cl_dist = cl_dist if cl_dist is not None else lambda x: 0
        self.transom_angle = transom_angle
        self.epsilon = epsilon
        self.ahr = ahr
        self.rho = rho
        self.nu = nu
        self.g = g
        self.n_sections = n_sections
        
        # State variables (updated by methods)
        self.z_wl = 0
        self.tau = 5  # Initial guess
        self.L_K = 0
        self.L_C = 0
        self.lambda_W = 0
        self.lcp = 0
        self.total_lift = 0
        self.skin_friction = np.zeros(3)
        self.spray_resistance = 0
        self.pressure_resistance = 0
        self.total_resistance = 0
        self.validity_flag = True
        
        # CPD regression coefficients from Eq. 4-38 (p. 87)
        self.cpd_coeffs = {
            'intercept': -0.0632,
            'Cv': 0.0812,
            'lambda': 0.187,
            'tau_deg': -0.0482,
            'beta_deg': -0.00581
        }
        
        # Validity limits from Table 6-17 (p. 128)
        self.validity_limits = {
            'lambda': (1.6, 4.0),
            'tau_deg': (2.0, 8.0),
            'beta_deg': (10.0, 30.0),
            'Cv': (2.0, 4.2)
        }
    
    def _interpolate_distribution(self, dist, x_vals):
        """Interpolate a distribution (beta, chine, or CL) at given x positions
        
        Args:
            dist: Distribution (list of [x, y] pairs or callable)
            x_vals: Array of x positions
            
        Returns:
            Interpolated values at x_vals
        """
        if callable(dist):
            return np.array([dist(x) for x in x_vals])
        elif isinstance(dist, (list, np.ndarray)):
            dist_arr = np.asarray(dist)
            f_interp = interpolate.interp1d(dist_arr[:, 0], dist_arr[:, 1], 
                                           kind='linear', fill_value='extrapolate',
                                           bounds_error=False)
            return f_interp(x_vals)
        else:
            raise ValueError("Distribution must be callable or list of [x, y] pairs")
    
    def _compute_wetted_limits(self, h, tau_deg):
        """Compute wetted surface boundaries using Wagner's theory (Sec 3.3/3.4)
        
        Args:
            h: Transom immersion (m)
            tau_deg: Trim angle (deg)
            
        Returns:
            L_K (keel wetted length), Xc (chine immersion point)
        """
        tau_rad = np.radians(tau_deg)
        
        # Wagner factor π/2 (p. 44)
        wagner_factor = np.pi / 2
        
        # Estimate keel wetted length from transom immersion and trim
        # L_K ≈ h / sin(τ) for small angles, corrected by Wagner factor
        if abs(tau_rad) < 1e-6:
            L_K = self.loa * 0.8  # Fallback for τ → 0
        else:
            L_K = min(h / np.sin(tau_rad) * wagner_factor, self.loa)
        
        # Chine immersion point (approximate as fraction of L_K)
        Xc = L_K * 0.9  # Simplified; thesis Sec 3.4 has more detailed treatment
        
        return max(L_K, 0.1), max(Xc, 0.1)
    
    def _get_sectional_properties(self, x_positions):
        """Get hull properties at section positions
        
        Args:
            x_positions: Array of longitudinal positions from stern (m)
            
        Returns:
            beta_x, beam_x, cl_x arrays
        """
        beta_x = self._interpolate_distribution(self.beta_dist, x_positions)
        beam_x = 2 * self._interpolate_distribution(self.chine_dist, x_positions)
        cl_x = self._interpolate_distribution(self.cl_dist, x_positions)
        return beta_x, beam_x, cl_x
    
    def _compute_cpd_regression(self, Cv, lambda_W, tau_deg, beta_deg):
        """Compute CPD position from Fridsma regression (Eq. 4-38, p. 87)
        
        CPD = a0 + a1*Cv + a2*λ + a3*τ + a4*β
        
        Returns:
            Adimensional CPD position (fraction of L_K from stern)
        """
        cpd = (self.cpd_coeffs['intercept'] + 
               self.cpd_coeffs['Cv'] * Cv +
               self.cpd_coeffs['lambda'] * lambda_W +
               self.cpd_coeffs['tau_deg'] * tau_deg +
               self.cpd_coeffs['beta_deg'] * beta_deg)
        return np.clip(cpd, 0.2, 0.8)  # Sanity bounds
    
    def _solve_sectional_lift_distribution(self, x_norm, b_norm, CL_BD, CPD_target, CLR):
        """Solve 4x4 linear system for CLSp(x) coefficients (Eq. 4-10, 4-11, p. 78)
        
        CLSp(x) = a0 + a12*x^(1/2) + a13*x^(1/3) + a1*x
        
        Boundary conditions:
        1. CLSp(0) = 1.0 (stagnation point)
        2. CLSp(1) = CLR (trailing edge)
        3. ∫CLSp(x)*b(x)dx = CL_BD (global lift conservation)
        4. ∫x*CLSp(x)*b(x)dx = CPD (center of pressure position)
        
        Args:
            x_norm: Normalized x positions [0, 1]
            b_norm: Normalized beam distribution b(x)/B_max
            CL_BD: Global dynamic lift coefficient (from Savitsky Eq. 2-16)
            CPD_target: Target CPD position (adimensional)
            CLR: Trailing edge lift coefficient
            
        Returns:
            coeffs [a0, a12, a13, a1], CLSp values at x_norm
        """
        # Build 4x4 system A*a = B
        # Using numerical integration (trapezoidal rule) for integral constraints
        
        n = len(x_norm)
        dx = x_norm[1] - x_norm[0] if n > 1 else 1.0
        
        # Row 1: CLSp(0) = 1.0 => a0 = 1.0
        row1 = [1.0, 0.0, 0.0, 0.0]
        B1 = 1.0
        
        # Row 2: CLSp(1) = CLR => a0 + a12 + a13 + a1 = CLR
        row2 = [1.0, 1.0, 1.0, 1.0]
        B2 = CLR
        
        # Rows 3-4: Integral constraints (numerical quadrature)
        # Compute basis function integrals weighted by b(x)
        sqrt_x = np.sqrt(x_norm)
        cbrt_x = np.cbrt(x_norm)
        
        # Integral of b(x)*[1, x^0.5, x^(1/3), x] dx
        int_b = np.trapz(b_norm, x_norm)
        int_b_sqrt = np.trapz(b_norm * sqrt_x, x_norm)
        int_b_cbrt = np.trapz(b_norm * cbrt_x, x_norm)
        int_b_x = np.trapz(b_norm * x_norm, x_norm)
        
        # Integral of x*b(x)*[1, x^0.5, x^(1/3), x] dx
        int_xb = np.trapz(x_norm * b_norm, x_norm)
        int_xb_sqrt = np.trapz(x_norm * b_norm * sqrt_x, x_norm)
        int_xb_cbrt = np.trapz(x_norm * b_norm * cbrt_x, x_norm)
        int_xb_x = np.trapz(x_norm * x_norm * b_norm, x_norm)
        
        row3 = [int_b, int_b_sqrt, int_b_cbrt, int_b_x]
        B3 = CL_BD
        
        row4 = [int_xb, int_xb_sqrt, int_xb_cbrt, int_xb_x]
        B4 = CPD_target * CL_BD  # CPD is moment-weighted
        
        A = np.array([row1, row2, row3, row4])
        B = np.array([B1, B2, B3, B4])
        
        # Solve with regularization for ill-conditioned systems
        try:
            cond = np.linalg.cond(A)
            if cond > 1e4:
                # Tikhonov regularization
                lambda_reg = 1e-6
                A = A + lambda_reg * np.eye(4)
                warnings.warn(f'Ill-conditioned 4x4 system (κ={cond:.2e}). Applied regularization.', stacklevel=2)
            
            coeffs = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # Fallback to least squares
            coeffs, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
            warnings.warn('4x4 system singular. Used least-squares solution.', stacklevel=2)
        
        # Evaluate CLSp(x)
        CLSp = (coeffs[0] + coeffs[1] * sqrt_x + coeffs[2] * cbrt_x + 
                coeffs[3] * x_norm)
        
        # Enforce monotonic decrease (physical constraint)
        if not np.all(np.diff(CLSp) <= 0):
            # Apply PCHIP-like monotonicity enforcement
            CLSp = np.minimum.accumulate(CLSp[::-1])[::-1]
        
        return coeffs, CLSp
    
    def _compute_savitsky_clbd(self, tau_deg, lambda_W, beta_deg, Cv):
        """Compute global dynamic lift coefficient using Savitsky Eq. 2-16
        
        This is used as input to the VPH method for lift conservation constraint.
        
        Returns:
            CL_BD (dynamic lift coefficient)
        """
        tau_rad = np.radians(tau_deg)
        beta_rad = np.radians(beta_deg)
        
        # Savitsky Eq. 2-16 (simplified form)
        # CL_BD = τ - 0.0065*β*Cλ^(-1/3) / (0.02 + 0.00155*Cλ + 0.00224*Cλ²)
        # Note: This is approximate; full Savitsky implementation would use tables
        
        C_lambda = lambda_W
        denom = 0.02 + 0.00155*C_lambda + 0.00224*C_lambda**2
        CL_BD = (tau_rad - 0.0065*beta_rad*C_lambda**(-1/3)) / denom
        
        return max(CL_BD, 0.01)  # Sanity bound
    
    def _compute_hydrostatic_lift(self, x_positions, h, tau_deg, cl_x, beta_x, beam_x):
        """Compute hydrostatic lift per section (Eq. 4-41 to 4-44, p. 88-89)
        
        Args:
            x_positions: Section positions from stern (m)
            h: Transom immersion (m)
            tau_deg: Trim angle (deg)
            cl_x: Centerline height at sections (m)
            beta_x: Deadrise at sections (deg)
            beam_x: Beam at sections (m)
            
        Returns:
            L_H (total hydrostatic lift), CPH (hydrostatic CP position)
        """
        tau_rad = np.radians(tau_deg)
        L_K = x_positions[-1] - x_positions[0] if len(x_positions) > 1 else 1.0
        
        # Flotation line: flut(X) = h - (L_K - X)*tan(τ) (Eq. 4-41)
        flut_line = h - (L_K - (x_positions - x_positions[0])) * np.tan(tau_rad)
        
        # Hydrostatic lift per unit length (Eq. 4-42)
        # l_H(X) = 0.5*ρ*g*[flut(X) - CL(X)]² / tan(β)
        draft_local = np.maximum(flut_line - cl_x, 0)
        beta_rad = np.radians(beta_x)
        
        # Avoid division by zero for β → 0
        tan_beta = np.tan(beta_rad)
        tan_beta = np.where(np.abs(tan_beta) < 1e-6, 1e-6, tan_beta)
        
        l_H = 0.5 * self.rho * self.g * draft_local**2 / tan_beta
        
        # Integrate for total hydrostatic lift (Eq. 4-43)
        L_H = np.trapz(l_H, x_positions)
        
        # Compute hydrostatic CP (Eq. 4-44)
        if L_H > 1e-6:
            CPH = np.trapz(x_positions * l_H, x_positions) / L_H
        else:
            CPH = L_K / 2  # Default to midship
        
        return L_H, CPH
    
    def get_geo_lengths(self, h=None, tau_deg=None):
        """Compute geometric quantities (wetted lengths, areas)
        
        Args:
            h: Transom immersion (m). Uses current self.z_wl if None.
            tau_deg: Trim angle (deg). Uses current self.tau if None.
            
        Updates:
            L_K, L_C, lambda_W
        """
        if h is None:
            h = self.z_wl
        if tau_deg is None:
            tau_deg = self.tau
        
        L_K, Xc = self._compute_wetted_limits(h, tau_deg)
        
        self.L_K = L_K
        self.L_C = L_K * 0.95  # Approximate mean wetted length
        self.lambda_W = self.L_C / self.beam
        
        return L_K, self.L_C, self.lambda_W
    
    def get_forces(self, h=None, tau_deg=None, runGeoLengths=True):
        """Compute all forces, moments, and resistance components
        
        This is the main computational method implementing the VPH algorithm:
        1. Discretize hull into sections
        2. Map each section to virtual prismatic hull
        3. Solve 4x4 system for CLSp(x) distribution
        4. Integrate dynamic and hydrostatic lift
        5. Compute resistance components
        
        Args:
            h: Transom immersion (m). Uses self.z_wl if None.
            tau_deg: Trim angle (deg). Uses self.tau if None.
            runGeoLengths: Whether to update geometric quantities first.
            
        Updates:
            total_lift, lcp, skin_friction, spray_resistance, 
            pressure_resistance, total_resistance
        """
        if h is None:
            h = self.z_wl
        if tau_deg is None:
            tau_deg = self.tau
        
        if runGeoLengths:
            self.get_geo_lengths(h, tau_deg)
        
        L_K = self.L_K
        tau_rad = np.radians(tau_deg)
        U = self.speed
        
        # Discretize into sections along wetted length
        x_positions = np.linspace(0, L_K, self.n_sections)
        x_norm = x_positions / L_K  # Normalized [0, 1]
        
        # Get sectional properties
        beta_x, beam_x, cl_x = self._get_sectional_properties(x_positions)
        b_norm = beam_x / self.beam
        
        # Compute influence parameters
        Cv = U / np.sqrt(self.g * self.beam)  # Speed coefficient
        lambda_W = self.lambda_W
        beta_eff = np.mean(beta_x)  # Effective deadrise for global params
        
        # Compute global dynamic lift coefficient (Savitsky Eq. 2-16)
        CL_BD = self._compute_savitsky_clbd(tau_deg, lambda_W, beta_eff, Cv)
        
        # Compute target CPD position (Ribeiro Eq. 4-38)
        CPD_target = self._compute_cpd_regression(Cv, lambda_W, tau_deg, beta_eff)
        
        # Trailing edge lift coefficient (dry transom assumption)
        # CLR = 1 - (V_R/U)², simplified to 0.3-0.5 for typical planing
        CLR = 0.4  # Typical value for dry transom at planing speeds
        
        # Solve for CLSp(x) distribution
        coeffs, CLSp = self._solve_sectional_lift_distribution(
            x_norm, b_norm, CL_BD, CPD_target, CLR
        )
        
        # Compute dynamic lift per unit length (Eq. 4-13)
        l_D = 0.5 * self.rho * U**2 * beam_x * CLSp
        
        # Integrate for total dynamic lift and CP
        L_D = np.trapz(l_D, x_positions)
        if L_D > 1e-6:
            CPD_dim = np.trapz(x_positions * l_D, x_positions) / L_D
        else:
            CPD_dim = L_K / 2
        
        # Compute hydrostatic lift (Eq. 4-41 to 4-44)
        L_H, CPH = self._compute_hydrostatic_lift(
            x_positions, h, tau_deg, cl_x, beta_x, beam_x
        )
        
        # Total lift and CP (Eq. 4-12, 4-15)
        self.total_lift = L_D + L_H
        if self.total_lift > 1e-6:
            self.lcp = (L_D * CPD_dim + L_H * CPH) / self.total_lift
        else:
            self.lcp = L_K / 2
        
        # Compute resistance components
        # 1. Friction drag (ITTC-57 line, Eq. 3-28, 3-29)
        Re = U * L_K / self.nu
        if Re > 1e5:
            C_f = 0.075 / (np.log10(Re) - 2)**2  # ITTC-57
        else:
            C_f = 1.328 / np.sqrt(Re)  # Laminar
        
        # Wetted surface area approximation
        S_wet = L_K * self.beam * (1 + 0.5 * lambda_W)
        D_f = 0.5 * self.rho * U**2 * S_wet * C_f
        
        # 2. Spray resistance (Eq. 4-25)
        # Spray area SR approximated from chine wetted length
        S_R = 0.1 * L_K * self.beam  # Simplified spray area
        delta_Cf = 0.0  # Roughness increment (can be added)
        R_spray = 0.5 * self.rho * U**2 * C_f * (1 + delta_Cf) * S_R
        
        # 3. Viscous pressure resistance (Eq. 4-27)
        R_p = self.total_lift * np.sin(tau_rad)
        
        # Total resistance
        self.total_resistance = D_f + R_spray + R_p
        self.skin_friction = np.array([D_f * np.cos(tau_rad), 
                                        D_f * np.sin(tau_rad), 
                                        0])
        self.spray_resistance = R_spray
        self.pressure_resistance = R_p
        
        # Check validity limits
        self.check_validity(Cv, lambda_W, tau_deg, beta_eff)
        
        return self.total_lift, self.lcp, self.total_resistance
    
    def check_validity(self, Cv=None, lambda_W=None, tau_deg=None, beta_deg=None):
        """Check if current state is within applicability limits (Sec 4.3.5, Table 6-17)
        
        Emits warnings if extrapolating beyond validated ranges.
        """
        if Cv is None:
            Cv = self.speed / np.sqrt(self.g * self.beam)
        if lambda_W is None:
            lambda_W = self.lambda_W
        if tau_deg is None:
            tau_deg = self.tau
        if beta_deg is None:
            beta_deg = np.mean(self._interpolate_distribution(
                self.beta_dist, np.linspace(0, self.L_K, 10)
            ))
        
        self.validity_flag = True
        warnings_list = []
        
        # Check each limit
        limits = [
            ('lambda', lambda_W, self.validity_limits['lambda']),
            ('tau_deg', tau_deg, self.validity_limits['tau_deg']),
            ('beta_deg', beta_deg, self.validity_limits['beta_deg']),
            ('Cv', Cv, self.validity_limits['Cv'])
        ]
        
        for name, value, (low, high) in limits:
            if value < low or value > high:
                self.validity_flag = False
                deviation = max(abs(value - low), abs(value - high)) / (high - low) * 100
                warnings_list.append(
                    f"{name}={value:.2f} outside [{low}, {high}] "
                    f"({deviation:.1f}% deviation)"
                )
        
        if warnings_list:
            warning_msg = "VPH method extrapolating: " + "; ".join(warnings_list)
            warnings.warn(warning_msg, stacklevel=2)
        
        return self.validity_flag
    
    def get_steady_trim(self, x0=[0.05, 3], tolF=1e-6, maxiter=50):
        """Solve for dynamic equilibrium (h, τ) using nonlinear solver
        
        Solves the coupled force/moment balance equations (Eq. 4-21, 4-22):
        R1(h,τ) = AR - BR*tan(ε) - Le = 0
        R2(h,τ) = Df*f + (AR - BR*tan(ε))*(LCP - LCG) + BR*[(XT - LCP)*tan(ε) + KG] = 0
        
        Args:
            x0: Initial guess [h (m), τ (deg)]. Defaults to [0.05, 3].
            tolF: Convergence tolerance. Defaults to 1e-6.
            maxiter: Maximum iterations. Defaults to 50.
            
        Updates:
            z_wl, tau, and all force/moment quantities
        """
        from scipy.optimize import root
        
        def residuals(x):
            h, tau_deg = x
            
            # Update state temporarily
            old_z_wl, old_tau = self.z_wl, self.tau
            self.z_wl, self.tau = h, tau_deg
            
            # Compute forces
            self.get_forces(h, tau_deg, runGeoLengths=True)
            
            # Restore state (solver will update if converged)
            self.z_wl, self.tau = old_z_wl, old_tau
            
            # Extract computed values
            LT = self.total_lift
            LCP = self.lcp
            Df = self.skin_friction[0]
            
            # Geometric parameters
            tau_rad = np.radians(tau_deg)
            eps_rad = np.radians(self.epsilon)
            KG = self.vcg
            LCG = self.lcg
            XT = 0  # Transom at x=0 (stern)
            
            # Auxiliary terms (Eq. 4-21 notation)
            W = self.weight
            AR = W * np.cos(tau_rad) - Df * np.sin(tau_rad)
            BR = Df * np.cos(tau_rad) + W * np.sin(tau_rad)
            
            # Required lift (Le) - for equilibrium, should equal LT
            Le = LT
            
            # Moment arm f
            KF = 0  # Simplified; would need thrust application point
            LCF = 0
            f = (KF - KG) * np.cos(tau_rad) + (LCF - LCG) * np.sin(tau_rad)
            
            # Residuals
            R1 = AR - BR * np.tan(eps_rad) - Le
            R2 = Df * f + (AR - BR * np.tan(eps_rad)) * (LCP - LCG) + \
                 BR * ((XT - LCP) * np.tan(eps_rad) + KG)
            
            return [R1, R2]
        
        # Solve nonlinear system
        sol = root(residuals, x0, method='hybr', tol=tolF, 
                   options={'maxfev': maxiter * 10})
        
        if sol.success:
            self.z_wl, self.tau = sol.x
            self.get_forces(self.z_wl, self.tau, runGeoLengths=True)
        else:
            warnings.warn(f'Equilibrium solver did not converge: {sol.message}', 
                         stacklevel=2)
        
        return self.z_wl, self.tau, sol.success
    
    def print_description(self, sigFigs=7):
        """Print formatted description of vessel and results"""
        volume = self.weight / (self.g * self.rho)
        Fn_beam = self.speed / np.sqrt(self.g * self.beam)
        
        print("=" * 60)
        print("VIRTUAL PRISMATIC HULL (VPH) METHOD - Ribeiro 2002")
        print("=" * 60)
        print(f"\n---VESSEL GEOMETRY---")
        print(f"LOA:              {self.loa:.{sigFigs}g} m")
        print(f"Beam:             {self.beam:.{sigFigs}g} m")
        print(f"Weight:           {self.weight:.{sigFigs}g} N")
        print(f"Mass:             {self.weight/self.g:.{sigFigs}g} kg")
        print(f"Volume:           {volume:.{sigFigs}g} m³")
        print(f"LCG:              {self.lcg:.{sigFigs}g} m from stern")
        print(f"VCG:              {self.vcg:.{sigFigs}g} m from keel")
        print(f"Fn (beam-based):  {Fn_beam:.{sigFigs}g}")
        
        print(f"\n---OPERATING CONDITION---")
        print(f"Speed:            {self.speed:.{sigFigs}g} m/s ({self.speed*1.944:.{sigFigs}g} knots)")
        print(f"Trim (τ):         {self.tau:.{sigFigs}g} deg")
        print(f"Transom immers.:  {self.z_wl:.{sigFigs}g} m")
        
        print(f"\n---WETTED GEOMETRY---")
        print(f"L_K (keel):       {self.L_K:.{sigFigs}g} m")
        print(f"L_C (mean):       {self.L_C:.{sigFigs}g} m")
        print(f"λ (L/B ratio):    {self.lambda_W:.{sigFigs}g}")
        
        print(f"\n---FORCES & MOMENTS---")
        print(f"Total Lift:       {self.total_lift:.{sigFigs}g} N")
        print(f"Lift/Weight:      {self.total_lift/self.weight:.{sigFigs}g}")
        print(f"LCP:              {self.lcp:.{sigFigs}g} m from stern")
        
        print(f"\n---RESISTANCE BREAKDOWN---")
        print(f"Friction:         {self.skin_friction[0]:.{sigFigs}g} N")
        print(f"Spray:            {self.spray_resistance:.{sigFigs}g} N")
        print(f"Pressure:         {self.pressure_resistance:.{sigFigs}g} N")
        print(f"Total R_T:        {self.total_resistance:.{sigFigs}g} N")
        print(f"L/D Ratio:        {self.total_lift/self.total_resistance:.{sigFigs}g}" if self.total_resistance > 0 else "L/D: N/A")
        
        print(f"\n---VALIDITY CHECK---")
        status = "✓ Within limits" if self.validity_flag else "⚠ EXTRAPOLATING"
        print(f"Status:           {status}")
        if not self.validity_flag:
            self.check_validity()
        
        print("=" * 60)
