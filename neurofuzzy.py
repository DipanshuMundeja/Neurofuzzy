import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

occupancy = np.random.rand(100) * 100
external_temp = np.random.rand(100) * 40

occupancy_level = ctrl.Antecedent(np.arange(0, 101, 1), 'occupancy_level')
external_temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'external_temperature')
cooling_power = ctrl.Consequent(np.arange(0, 101, 1), 'cooling_power')

occupancy_level['low'] = fuzz.trimf(occupancy_level.universe, [0, 0, 50])
occupancy_level['high'] = fuzz.trimf(occupancy_level.universe, [25, 100, 100])
external_temperature['normal'] = fuzz.trimf(external_temperature.universe, [0, 0, 25])
external_temperature['hot'] = fuzz.trimf(external_temperature.universe, [15, 40, 40])
cooling_power['low'] = fuzz.trimf(cooling_power.universe, [0, 0, 50])
cooling_power['high'] = fuzz.trimf(cooling_power.universe, [25, 100, 100])


rule1 = ctrl.Rule(occupancy_level['low'] | external_temperature['normal'], cooling_power['low'])
rule2 = ctrl.Rule(occupancy_level['high'] & external_temperature['hot'], cooling_power['high'])

cooling_ctrl = ctrl.ControlSystem([rule1, rule2])
cooling_simulation = ctrl.ControlSystemSimulation(cooling_ctrl)

for i in range(len(occupancy)):
    cooling_simulation.input['occupancy_level'] = occupancy[i]
    cooling_simulation.input['external_temperature'] = external_temp[i]
    cooling_simulation.compute()
    print(f"Occupancy: {occupancy[i]}, External Temp: {external_temp[i]}, Cooling Power: {cooling_simulation.output['cooling_power']}")
