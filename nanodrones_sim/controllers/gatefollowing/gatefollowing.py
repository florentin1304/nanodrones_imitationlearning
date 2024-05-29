import sys
sys.path.append("..")
sys.path.append("../../..") #home
import numpy as np
import pandas as pd
import scipy
from bayes_opt import BayesianOptimization as BO
from bayes_opt import SequentialDomainReductionTransformer
from controller_utils.gate_follower_supervisor import GateFollowerSupervisor

def parameter_analysis():
    def min_func(gf, smoothing, ax_max, ax_min, ay_max):
        x = (smoothing, ax_max, ax_min, ay_max)
        print("Evaluating x: ", x)
        config = {
            'recorder':{
                'mode': 'on',
                'save_dir': 'data',
                'save_img': bool(False)
            },

            'pathplanner':{
                'smoothing_factor': smoothing,
                'velocity_profiler_config': {
                    'ax_max': ax_max,
                    'ax_min': ax_min,
                    'ay_max': ay_max
                }
            }
        }

        statistics = {'time': [], 'completed':[] ,'ctrl_error':[], 'traj_error':[]}
        for rad in [6]:
            for frac in [1.0, 0.66, 0.33]:
                traj_conf = {
                    'trajectory_generator':{
                        'traj_type': 'ellipse',
                        'traj_conf': {
                            'radius': rad, 
                            'other_radius_frac': frac,
                            'shift_left': False
                        }
                    }
                }
                config.update(traj_conf)
                
                # Simulate
                gf.reset(config=config)
                status = 'flying'
                while status == 'flying':
                    status = gf.step()
                
                # Get data
                df = gf.recorder.get_data_df()
                statistics['time'].append(df['sim_time'].iloc[-1] - df['sim_time'].iloc[0])
                statistics['completed'].append(status == 'finished')
                tot_err = 0
                for i, var in enumerate(['roll_ctrl_error', 'pitch_ctrl_error', 'yaw_rate_ctrl_error']):
                    tot_err += abs(df[var]).mean()
                statistics['ctrl_error'] = tot_err

                df['dist_from_proj'] = np.linalg.norm(
                    df[['x','y','z']].to_numpy() - df[['x_proj','y_proj','z_proj']].to_numpy(),
                    axis=1
                )
                statistics['traj_error'] = df['dist_from_proj'].mean()
        
        score = np.mean(statistics['time']) + \
                100*np.sum([1 if not(a) else 0 for a in statistics['completed']]) + \
                20 * np.mean(statistics['ctrl_error']) + \
                20 * np.mean(statistics['traj_error'])
        print(f"FINAL RESULT ({x=}): ", score)
        
        return score
    
    gf = GateFollowerSupervisor(display_path=True)
    func_to_max = lambda smoothing, ax_max, ax_min, ay_max: -min_func(gf, smoothing, ax_max, ax_min, ay_max)
    pbounds = {'smoothing': (0.85, 0.95),
                'ax_max': (1, 5),
               'ax_min': (1, 5),
               'ay_max': (0.5, 5)}
    
    min_window = {'smoothing': 0.05,
                'ax_max': 1,
                'ax_min': 1,
                'ay_max': 1}
    min_window_list = [item[1] for item in sorted(min_window.items(), key=lambda x: x[0])]
    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=min_window_list)

    optimizer = BO(
        f=func_to_max,
        pbounds=pbounds,
        random_state=0,
        verbose=1,
        allow_duplicate_points=True,
        bounds_transformer=bounds_transformer
    )

    optimizer.probe(
        params={'smoothing': 0.9,
                'ax_max': 3.0,
               'ax_min': 3.0,
               'ay_max': 1.0},
        lazy=True,
    )

    print('Start optimization...')
    optimizer.maximize(
        init_points=0,
        n_iter=50,
    )

    res = optimizer.max
    

    print(f"FINAL RESULT: ", res)
    with open("xres.txt", 'w') as f:
        f.write(str(res))


if __name__ == "__main__":
    print("Starting simulation")
    if False:
        parameter_analysis()
    else:
        gf = GateFollowerSupervisor(display_path=True)

        for _ in range(100):
            config = {
                'recorder':{
                    'mode': 'on',
                    'save_dir': 'data',
                    'save_img': bool(False)
                },

                # 'trajectory_generator':{
                #     'traj_type': 'ellipse',
                #     'traj_conf': {
                #         'radius': np.random.uniform(3, 7), 
                #         'other_radius_frac': np.random.uniform(0.4, 1.6),
                #         'shift_left': bool(np.random.choice([False, True]))
                #     }
                # },
                'trajectory_generator':{
                    'traj_type': 'csv'
                },

                'pathplanner':{
                    'target_distance': 0.4,
                    'smoothing_factor': 0.91,
                    'velocity_profiler_config': {
                        'ax_max': 3,
                        'ax_min': 2,
                        'ay_max': 0.9
                    }
                }
            }

            gf.reset(config=config)
            status = 'flying'
            while status == 'flying':
                status = gf.step()

            if status == 'finished':
                gf.recorder.save_data()
            