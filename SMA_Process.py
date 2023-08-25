#%%
import numpy as np

from CADETProcess.processModel import ComponentSystem
from CADETProcess.processModel import StericMassAction, NoBinding
from CADETProcess.processModel import Inlet, LumpedRateModelWithPores, LumpedRateModelWithoutPores, TubularReactor,Outlet
from CADETProcess.processModel import FlowSheet
from CADETProcess.processModel import Process
from pathlib import Path
%matplotlib inline
# Component System
component_system = ComponentSystem()
component_system.add_component('Salt')
Prot_1='BSA'
Prot_2='LYZ'
Prot_3='CYC'
c0_s=5
component_system.add_component(Prot_1)
component_system.add_component(Prot_2)
component_system.add_component(Prot_3)

# Binding Model
binding_model = StericMassAction(component_system, name='SMA')
binding_model.is_kinetic = False
binding_model.adsorption_rate = [0.0, 0.00021, 0.1329, 0.229]
binding_model.desorption_rate = [1, 1, 1, 1]
binding_model.characteristic_charge = [0.0, 18.2, 7.447, 9.384]
binding_model.steric_factor = [0.0, 60, 15, 15]
binding_model.capacity = 1200.0
binding_model.reference_liquid_phase_conc=1000
binding_model.reference_solid_phase_conc=1200

# Unit Operations
inlet = Inlet(component_system, name='inlet')
inlet.flow_rate = 8.333e-9

column = LumpedRateModelWithPores(component_system, name='column')
column.binding_model = binding_model

column.length = 0.1
column.diameter = 0.0077357778
column.bed_porosity = 0.37
column.particle_radius = 3.5e-5/2
column.particle_porosity = 0.75
column.axial_dispersion = 5.75e-8
column.film_diffusion = [1e-4,1.1e-7,5e-6,5e-6]
#column.pore_diffusion = [7e-10, 6.07e-11, 6.07e-11, 6.07e-11]
#column.surface_diffusion = column.n_bound_states*[0.0]

column.c = [c0_s, 0, 0, 0]
column.cp = [c0_s, 0, 0, 0]
column.q = [binding_model.capacity, 0, 0, 0]

outlet = Outlet(component_system, name='outlet')
"""
Tubing_Front=TubularReactor(component_system, name='Tubing_Front')
Tubing_Front.c = [c0_s, 0, 0, 0]
Tubing_Front.axial_dispersion = 5.75e-8
Tubing_Front.length = 0.05
Tubing_Front.diameter = 0.005
#Tubing_Front.total_porosity = 1
"""
Tubing =StericMassAction(component_system, name='No_binding')
Tubing.is_kinetic = False
Tubing.adsorption_rate = [0.0, 0.000, 0., 0.]
Tubing.desorption_rate = [1, 1, 1, 1]
Tubing.characteristic_charge = [0.0, 18.2, 7.447, 9.384]
Tubing.steric_factor = [0.0, 60, 15, 15]
Tubing.capacity = 0#1200.0
Tubing_Front=LumpedRateModelWithoutPores(component_system, name='Tubing_Front')
Tubing_Front.c = [c0_s, 0, 0, 0]
Tubing_Front.axial_dispersion = 5.75e-8
Tubing_Front.length = 0.05
Tubing_Front.diameter = 0.005
Tubing_Front.total_porosity = 1
Tubing_Front.binding_model=Tubing

# Flow Sheet
flow_sheet = FlowSheet(component_system)

flow_sheet.add_unit(inlet,feed_inlet=True,eluent_inlet=True)
flow_sheet.add_unit(Tubing_Front)
flow_sheet.add_unit(column)
flow_sheet.add_unit(outlet, product_outlet=True)

flow_sheet.add_connection(inlet, Tubing_Front)
flow_sheet.add_connection(Tubing_Front, column)
flow_sheet.add_connection(column, outlet)

# Process
process = Process(flow_sheet, 'SMA')
process.cycle_time = 110*60
t_flush=20*60

load_duration = 9
t_gradient_start = 90.0
gradient_duration = process.cycle_time - t_gradient_start-t_flush

c_load = np.array([c0_s, 1.0, 1.0, 1.0])
c_wash = np.array([c0_s, 0.0, 0.0, 0.0])
c_elute = np.array([1000.0, 0.0, 0.0, 0.0])
gradient_slope = (c_elute - c_wash)/gradient_duration
c_gradient_poly = np.array(list(zip(c_wash, gradient_slope)))
c_gradient_poly[0][1]=0.4

process.add_event('load', 'flow_sheet.inlet.c', c_load)
process.add_event('wash', 'flow_sheet.inlet.c',  c_wash, load_duration)
process.add_event('grad_start', 'flow_sheet.inlet.c', c_gradient_poly[0][1],indices=(0,1), time=t_gradient_start)
print(inlet.c)
process.add_event('Flush','flow_sheet.inlet.c',c_elute,t_gradient_start+gradient_duration)
#print(inlet.c)
#
from CADETProcess.simulator import Cadet
process_simulator = Cadet(install_path=Path.home() / "Cadet" / "cadet"/"bin"/"cadet-cli.exe")

simulation_results = process_simulator.simulate(process)

from CADETProcess.plotting import SecondaryAxis
sec = SecondaryAxis()
sec.components = ['Salt']
sec.y_label = '$c_{salt}$'
_=simulation_results.solution.column.outlet.plot(secondary_axis=sec)


from CADETProcess.fractionation import FractionationOptimizer
fractionation_optimizer = FractionationOptimizer()
fractionator = fractionation_optimizer.optimize_fractionation(simulation_results, components=[Prot_1,Prot_2,Prot_3], purity_required=[0.85,0., 0.])
#

_ = fractionator.plot_fraction_signal()
print(fractionator.performance)

#%%
from CADETProcess.optimization import OptimizationProblem
optimization_problem = OptimizationProblem('SMA', use_diskcache=False)

optimization_problem.add_evaluation_object(process)

var=optimization_problem.add_variable('Gradient_Slope',parameter_path='grad_start.state',lb=0.1, ub=0.4)
var.value=0.2#c_gradient_poly[0][1]
print(inlet.c)

#optimization_problem.add_variable('flow_sheet.inlet.c', lb=0.1, ub=0.3)
#optimization_problem.add_variable('feed_duration.time', lb=10, ub=300)
process_simulator.evaluate_stationarity = False#True

optimization_problem.add_evaluator(process_simulator)
optimization_problem.add_evaluator(
    fractionation_optimizer,
    kwargs={
        'components': [Prot_1,Prot_2,Prot_3],
        'purity_required': [0.85,0,0],
        'ignore_failed': False,
        'allow_empty_fractions': False
    }
)
def callback(fractionation, individual, evaluation_object, callbacks_dir):
    fractionation.plot_fraction_signal(
        file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_fractionation.png',
        show=False
    )



optimization_problem.add_callback(
    callback, requires=[process_simulator, fractionation_optimizer]
)
from CADETProcess.performance import Mass
ranking = [1, 0, 0]
performance = Mass(ranking=ranking)

optimization_problem.add_objective(
    performance, requires=[process_simulator, fractionation_optimizer]
)

#print(c_gradient_poly[0][1])
#print(' führt zu: ')
#print(fractionator.performance)
var.value=0.3
print('0.3 führt zu: ')
print(optimization_problem.evaluate_objectives([0.3],True))
var.value=0.15
print('0.15 führt zu: ')
print(optimization_problem.evaluate_objectives([0.15],True))
"""print('0.4 führt zu: ')
print(optimization_problem.evaluate_objectives([0.40]))
print('0.1 führt zu: ')
print(optimization_problem.evaluate_objectives([0.10]))"""
from CADETProcess.optimization import U_NSGA3
optimizer = U_NSGA3()
optimizer.n_max_gen=3
optimizer.pop_size=10
"""
"""

#%%

print('Start Optimizing...')
results = optimizer.optimize(
    optimization_problem,
    use_checkpoint=False,
)

# %%
