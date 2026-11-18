import SofaRuntime
import Sofa.Core
root = Sofa.Core.Node()
import math
import numpy as np
from scipy import signal
import pickle
import os
import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial
import DeLaN_model_v4 as delan   

# =========================================================
# 0. CONFIGURATION FLAG -- choose which adapted model to use
# =========================================================
METHOD = "subspace"      # "subspace" or "finetune"
SUBSPACE_RANK = 2         # only used if METHOD == "subspace"

RESULTS_DIR = "./models"

# =========================================================
# 1. Load hyperparameters (from either model -- they're shared)
# =========================================================
if METHOD == "subspace":
    model_path = f"{RESULTS_DIR}/subspace_rank{SUBSPACE_RANK}.jax"
elif METHOD == "finetune":
    model_path = f"{RESULTS_DIR}/fine_tune.jax"
else:
    raise ValueError(f"Unknown METHOD: {METHOD}")

with open(model_path, "rb") as f:
    data = pickle.load(f)

hyper = data["hyper"]
params = data["params"]

print(f"[Controller] Loaded method = '{METHOD}'"
      + (f" (rank={SUBSPACE_RANK})" if METHOD == "subspace" else "")
      + f" from: {model_path}")

# =========================================================
# 2. Build apply functions (architecture is identical for both methods)
# =========================================================
activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

lagrangian_fn = hk.transform(partial(
    delan.structured_lagrangian_fn,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
    epsilon=hyper['diagonal_epsilon'],
    shift=hyper['diagonal_shift'],
))

dissipative_fn = hk.transform(partial(
    delan.dissipative_matrix,
    n_dof=hyper['n_dof'],
    shape=(5,) * 3,
    activation=activations[hyper['activation2']]
))

potential_fn = hk.transform(partial(
    delan.potential_energy_fn,
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
))

input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))

V = lambda q: potential_fn.apply(params["lagrangian"], None, q).squeeze()

input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))

lagrangian = lagrangian_fn.apply
input_mat = input_mat_fn.apply

def AL(q):
    return input_mat(params['input_transform'], None, q)

def GL(q):
    return jax.grad(V)(q)

AL_jit = jax.jit(AL)
GL_jit = jax.jit(GL)
scale_factor = 10000

def feedforward_control(qd):
    Al = AL_jit(qd)
    Gl = GL_jit(qd)
    u_ff = np.linalg.pinv(np.array(Al)) @ np.array(Gl)/scale_factor
    return float(np.array(u_ff).squeeze())

def feedback_control(q, qd, dq, Kp=0.5, Kd=0.01):
    Al = AL_jit(qd)
    u_fb = Al*(Kp * (qd-q) - Kd * dq)/scale_factor
    return u_fb


def calculate_angle(origin, tip, angle0=0.0, dt=0.01):
    dx = tip[0] - origin[0]
    dz = tip[2] - origin[2]
    angle = -math.atan2(dx, dz)
    dangle = (angle - angle0) / dt if dt > 1e-9 else 0.0
    return angle, dangle




class CylinderController(Sofa.Core.Controller):
    n_States = 4
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.node = kwargs['node']
        self.effectorMO = kwargs['effectorMO']
        self.originMO = kwargs['originMO']
        self.q1PointMO = kwargs['q1PointMO']
        self.q2PointMO = kwargs['q2PointMO']
        self.q3PointMO = kwargs['q3PointMO']
        self.pressureValue = kwargs['pressure']

        self.x_prev = np.zeros(4, dtype=float)

        # Output filename now tags the method used, so results don't get overwritten
        out_name = f"regulation_{METHOD}" + (f"_rank{SUBSPACE_RANK}" if METHOD == "subspace" else "") + ".txt"
        self.file = open(out_name, "w")
        self.file.write(
            f"# method={METHOD}"
            + (f" rank={SUBSPACE_RANK}" if METHOD == "subspace" else "")
            + "\n"
            "# time[s] u[MPa] "
            "x0[deg] x1[deg] x2[deg] x3[deg] "
            "dx0[deg/s] dx1[deg/s] dx2[deg/s] dx3[deg/s]\n"
        )
        print(f"[Controller] Logging results to: {out_name}")

    def onAnimateEndEvent(self, event):
        time = float(self.node.time.value)

        tip = self.effectorMO.position.value[0]
        origin = self.originMO.position.value[0]

        # Convert SOFA mm -> meters, to match training data units

        ex_val, ey_val, ez_val = tip[0], tip[1], tip[2]

        q1_point = self.q1PointMO.position.value[0]
        q2_point = self.q2PointMO.position.value[0]
        q3_point = self.q3PointMO.position.value[0]

        rad2deg = 180/np.pi
        dt = float(self.node.dt.value)

        x0, dx0 = calculate_angle(origin, q1_point, angle0=self.x_prev[0], dt=dt)
        x1, dx1 = calculate_angle(origin, q2_point, angle0=self.x_prev[1], dt=dt)
        x2, dx2 = calculate_angle(origin, q3_point, angle0=self.x_prev[2], dt=dt)
        x3, dx3 = calculate_angle(origin, tip,      angle0=self.x_prev[3], dt=dt)
        q_ref = np.array([0.62790, 0.07604, -0.75411, np.deg2rad(34.99)], dtype=float)

        if time <= dt:
            dx0 = dx1 = dx2 = dx3 = 0.0

        self.x_prev[:] = [x0, x1, x2, x3]

        q = np.array([ex_val, ey_val, ez_val, x3], dtype=float)
        dq = np.array([0.0, 0.0, 0.0, dx3], dtype=float)  # velocities for x,y,z optional; see note below

        u_fb = feedback_control(q, q_ref, dq)
        u_ff = feedforward_control(q_ref)
        pressure_cmd = u_ff
        pressure_cmd = float(np.asarray(pressure_cmd).squeeze())

        try:
            self.pressureValue.value[0] = pressure_cmd
        except TypeError:
            self.pressureValue.value = pressure_cmd

        print(
            f"[{METHOD}] Time: {time:.3f} s | "
            f"Pressure cmd: {pressure_cmd:.6f} MPa | "
            f"Angles: [{x0*rad2deg:.2f}, {x1*rad2deg:.2f}, {x2*rad2deg:.2f}, {x3*rad2deg:.2f}] deg | "
            f"dAngles: [{dx0*rad2deg:.2f}, {dx1*rad2deg:.2f}, {dx2*rad2deg:.2f}, {dx3*rad2deg:.2f}] deg/s"
        )

        self.file.write(
            f"{time:.6f} {pressure_cmd:.6e} "
            f"{x0*rad2deg:.6f} {x1*rad2deg:.6f} {x2*rad2deg:.6f} {x3*rad2deg:.6f} "
            f"{dx0*rad2deg:.6f} {dx1*rad2deg:.6f} {dx2*rad2deg:.6f} {dx3*rad2deg:.6f}\n"
        )
        self.file.flush()

    def cleanup(self):
        self.file.close()


def createScene(rootNode):

    ## UNIT OF MEASURE : LENGHT [mm] MASS [tonne] TIME [s] PRESSURE/STRESS [MPa]

    rootNode.addObject('VisualStyle', displayFlags='showVisualModels hideBehaviorModels hideCollisionModels hideBoundingCollisionModels hideForceFields hideInteractionForceFields hideWireframe')
    
    rootNode.addObject('RequiredPlugin', name='SofaPython3')
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.Engine.Select")
    rootNode.addObject('RequiredPlugin', name="Sofa.Component.IO.Mesh")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Iterative")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mapping.Linear")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Mass")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Forward")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Setting")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.FEM.HyperElastic")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.FEM.Elastic")	
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.SolidMechanics.Spring")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.StateContainer")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Dynamic")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Visual")
    rootNode.addObject("RequiredPlugin", name="Sofa.GL.Component.Rendering3D")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.AnimationLoop")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Lagrangian.Solver")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.MechanicalLoad")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.LinearSolver.Direct")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Constraint.Lagrangian.Correction")
    rootNode.addObject("RequiredPlugin", name = "Sofa.Component.Constraint.Projective")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.ODESolver.Backward")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Algorithm")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection.Intersection")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Response.Contact")
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Topology.Container.Constant")
    rootNode.addObject("RequiredPlugin", name="SoftRobots")  
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Geometry")



    rootNode.gravity=[9810,0,0]
    rootNode.dt = (0.005)
    rootNode.name = 'rootNode'
    rootNode.addObject('FreeMotionAnimationLoop')
    # rootNode.addObject('GenericConstraintSolver', tolerance=1e-12, maxIterations=10000, computeConstraintForces = True)
    rootNode.addObject(
    "BlockGaussSeidelConstraintSolver",
    maxIterations=10000,
    tolerance=1e-8, 
    )
    # rootNode.addObject('DefaultPipeline')
    rootNode.addObject('CollisionPipeline')

    rootNode.addObject('BruteForceBroadPhase')
    rootNode.addObject('BVHNarrowPhase')
    # rootNode.addObject('DefaultContactManager', response='FrictionContactConstraint', responseParams='mu=0.6')
    # rootNode.addObject('ContactManager', response='FrictionContactConstraint', responseParams='mu=0.6')
    rootNode.addObject("RequiredPlugin", name="Sofa.Component.Collision.Detection")


    rootNode.addObject('LocalMinDistance', name='Proximity', alarmDistance = 2, contactDistance=1, angleCone=0.0)
    rootNode.addObject('OglSceneFrame', style='Arrows', alignment='TopRight')


##  Material DBPM-F5000- R0.3  (BLUE)
    finger = rootNode.addChild('finger')

    #finger.addObject('EulerImplicitSolver', name="Solver_Ogden",rayleighMass = 0.1, rayleighStiffness = 0.1, firstOrder = False, trapezoidalScheme = False) 
    finger.addObject('EulerImplicitSolver', name="Solver_Mooney",rayleighMass = 0.0, rayleighStiffness = 0.0, firstOrder = False, trapezoidalScheme = False) 

    # With 0.0 at rayleighStiffness and rayleighMass the ODE solver tries to preserve the dynamics (no artificial dumping added). The euler solver is extremely useful if you want involve contacts (it is the faster and most stable).
    # OGDEN is an exponential model of hyperelasticity, with zero artificial dumping is not stable, Mooney Rivlin is polynomial, so is more stable in simulation in general, also with zero artificial dumping (recommended) 

    finger.addObject('SparseLDLSolver', name='directSolver')

    finger.addObject('MeshVTKLoader', name='loader', filename='design/finger1.vtu', translation = [0, 0.0, 0])
    finger.addObject('MechanicalObject', name='tetras', template='Vec3d', src = '@loader')
    finger.addObject('TetrahedronSetTopologyContainer', name="topo", src ='@loader')
    finger.addObject('TetrahedronSetTopologyModifier' ,  name="Modifier")
    finger.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3d" ,name="GeomAlgo")

    finger.addObject('UniformMass', totalMass="28e-6", src = '@topo') # usually the finger should wight 28-30 g
    finger.addObject('BoxROI', name='boxROI1', box=[-12, -5, -2, 10,25, 5], drawBoxes=True)
    finger.addObject('FixedConstraint', indices = '@boxROI1.indices')



# OGDEN COEFFICIENT

#     C1 = 0.214210367 #[MPa]
#     alpha = 1.61052759
#     D1_ogden = 1.16707703 #[MPa^-1] ---> D1 = 2/K --> K is the bulk modulus
#     K_ogden = 2/D1_ogden

# # MOONEY RIVLIN COEFFICIENT
#     C10 = 0.08859383074 #[MPa]
#     C01 = 0.01365494839 #[MPa]
#     D1_mooney = 1.22250848 #[MPa^-1] ---> D1 = 2/K --> K is the bulk modulus
#     K_mooney = 2/D1_mooney


#     ## LINEAR ELASTIC COEFFICIENT
#     E = 0.6988 #MPa
#     nu = 0.44

    # ================= ROBOT 1 =================
    # C1 = 0.214210367
    # alpha = 1.61052759
    # D1_ogden = 1.16707703
    # K_ogden = 2 / D1_ogden

    # C10 = 0.08859383074
    # C01 = 0.01365494839
    # D1_mooney = 1.22250848
    # K_mooney = 2 / D1_mooney

    # E = 0.6988
    # nu = 0.44


    # ================= ROBOT 2 =================
    # C1 = 0.1850
    # alpha = 1.48
    # D1_ogden = 1.3500
    # K_ogden = 2 / D1_ogden

    # C10 = 0.0740
    # C01 = 0.0115
    # D1_mooney = 1.4200
    # K_mooney = 2 / D1_mooney

    # E = 0.6100
    # nu = 0.43


    # ================= ROBOT 3 =================
    # C1 = 0.2000
    # alpha = 1.58
    # D1_ogden = 1.2300
    # K_ogden = 2 / D1_ogden

    # C10 = 0.0835
    # C01 = 0.0128
    # D1_mooney = 1.2850
    # K_mooney = 2 / D1_mooney

    # E = 0.6600
    # nu = 0.445


    # ================= ROBOT 4 =================
    # C1 = 0.2300
    # alpha = 1.72
    # D1_ogden = 1.0500
    # K_ogden = 2 / D1_ogden

    # C10 = 0.0960
    # C01 = 0.0152
    # D1_mooney = 1.1000
    # K_mooney = 2 / D1_mooney

    # E = 0.7600
    # nu = 0.455


    # ================= ROBOT 5 =================
    C1 = 0.2550
    alpha = 1.88
    D1_ogden = 0.9200
    K_ogden = 2 / D1_ogden

    C10 = 0.1080
    C01 = 0.0175
    D1_mooney = 0.9700
    K_mooney = 2 / D1_mooney

    E = 0.8500
    nu = 0.465
        
    finger.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', src ='@topo', method = 'large', youngModulus = E, poissonRatio = nu) ## Linear Elastic	

    finger.addObject('LinearSolverConstraintCorrection', linearSolver='@directSolver')



    cavity = finger.addChild('cavity')
    cavity.addObject('MeshOBJLoader', name='cavityLoader', filename='design/BSPA_New_Design_Inner.obj')
    cavity.addObject('MeshTopology', src='@cavityLoader', name='cavityMesh')
    cavity.addObject('MechanicalObject', name='cavity', rotation = [0, 0 , 0])
    cavity.addObject('SurfacePressureConstraint', name='SurfacePressureConstraint', template='Vec3d', value= 0, triangles='@cavityMesh.triangles', valueType='pressure')
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)


    effector = finger.addChild('effector')
    effectorMO = effector.addObject('MechanicalObject', name='effectorMO', template='Vec3d', position=[10.0,8.0,-100.], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    effector.addObject('BarycentricMapping')

    origin = finger.addChild('origin')
    originMO = origin.addObject('MechanicalObject', name='effectorMO', template='Vec3d', position=[10.0,8.0,0.0], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    origin.addObject('BarycentricMapping')


    # I will add more points to define a more appropriate state.
    q2_point = finger.addChild('q2_point')
    q2_pointMO = q2_point.addObject('MechanicalObject', name='q2_pointMO', template='Vec3d', position=[10.0,8.0,-50.], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    q2_point.addObject('BarycentricMapping')

    q1_point = finger.addChild('q1_point')
    q1_pointMO = q1_point.addObject('MechanicalObject', name='q1_pointMO', template='Vec3d', position=[10.0,8.0,-25.], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    q1_point.addObject('BarycentricMapping')

    q3_point = finger.addChild('q3_point')
    q3_pointMO = q3_point.addObject('MechanicalObject', name='q3_pointMO', template='Vec3d', position=[10.0,8.0,-75.], showObject = True, showObjectScale=4, drawMode=2, showColor=[1., 0, 0., 1.])
    q3_point.addObject('BarycentricMapping')




    # rootNode.addObject(CylinderController(node=rootNode, pos3 = rootNode.finger.tetras.position.value, pressure = rootNode.finger.cavity.SurfacePressureConstraint, effectorMO=effectorMO, originMO=originMO))
    # rootNode.addObject(CylinderController(node=rootNode, pos3 = rootNode.finger.tetras.position.value, pressure = rootNode.finger.cavity.SurfacePressureConstraint, effectorMO=effectorMO, originMO=originMO)) #, q1PointMO=q1_pointMO, q2PointMO=q2_pointMO, q3PointMO=q3_pointMO))
    rootNode.addObject(CylinderController(node=rootNode, pos3 = rootNode.finger.tetras.position.value, pressure = rootNode.finger.cavity.SurfacePressureConstraint, effectorMO=effectorMO, originMO=originMO, q1PointMO=q1_pointMO, q2PointMO=q2_pointMO, q3PointMO=q3_pointMO))

    modelVisu = finger.addChild('visu')
    modelVisu.addObject('MeshOBJLoader', name='loader', filename='design/BSPA_New_Design_Outer.obj', translation = [0.0,0.0,0.0])
    modelVisu.addObject('OglModel', src='@loader', color=[1,0.64,0,1]) # color=[0,0.64,0,1]
    modelVisu.addObject('BarycentricMapping')



    return rootNode