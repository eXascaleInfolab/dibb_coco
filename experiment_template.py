import warnings

# Remove warning from cocoex
warnings.filterwarnings('ignore', message='"is" with a literal. Did you mean "=="?')

from tqdm import tqdm
from datetime import datetime
import os
import time
import numpy as np
import cocoex
import platform
import cma
from brutelogger import BruteLogger
import ray
from dibb import DiBB
import traceback

"""
To run different experiments, make a copy of this file and adapt the configuration parameters.
You can either run one experiment per file or multiple experiments if enough machines are 
available. The Experiment class below is meant to run only for one specific dimension and number
of blocks, so if you want to run problems of multiple dimensions at the same time, add them to the
list exp_config[experiments].
"""

class Experiment:
    def __init__(self, name, dimension, nblocks, head_node_ip, progress_bar_color='#FFFFFF'):
        self.name = name
        self.dimension = dimension
        self.head_node_ip = head_node_ip
        self.progress_bar_color = progress_bar_color
        self.nblocks = nblocks

exp_config = {
    # Where to store the logs
    'log_dir': '/home/glf/logs',
    # If true, notify will send notifications on experiment start and end
    'use_notify': False,
    'use_cma_ipop': True,
    'use_dibb_ipop': False,
    # List of experiments (one dimension per experiment)
    'experiments': [
        Experiment(
            name='ls_40d_2b',
            dimension=40,
            head_node_ip='134.21.220.201',
            nblocks=2,
        )
    ],
    # Function that generates a new initial individual given a coco problem
    'init_ind_generator': lambda problem: np.random.uniform(-4, 4, problem.dimension)
}

coco_config = {
    # Which suite to use in the experiment
    'suite': 'bbob-largescale',
    # Evaluation budget = budget_multiplier * dimension
    'budget_multiplier': 1e4,
    # Either 'error', 'warning', 'info' or 'debug'
    'log_level': 'warning',
    # Which problem indices should be loaded 
    'problem_indices': [*range(0, 75), *range(135, 210)],  # Separable (f1-f5) and ill-conditioned (f10-f14) problems
}

dibb_config = {
    'verbose': False,
    'optimizer': 'cma-ipop' if exp_config['use_dibb_ipop'] else'cma',
    'optimizer_options': {
        'init_sigma': 2,
        'verbose': -9,
    },
    'nfitness_evaluators': 0,
    'print_fn': None,
}

ray_config = {
    'configure_logging': False,
    'logging_level': 50,
    'address': 'auto',
}

script_name = os.path.basename(__file__).strip('.py')
fname = f'log_{script_name}_{datetime.now().strftime("%y%m%d_%H%M")}.txt'
BruteLogger.save_stdout_to_file(path=exp_config['log_dir'], fname=fname, encoding='utf8', also_stderr=True)

assert not (exp_config['use_cma_ipop'] and exp_config['use_dibb_ipop']), 'It is not recommended to use both CMA-IPOP and DIBB-IPOP at the same time'

if exp_config['use_notify']:
    from notify_run import Notify
    notify = Notify()
    
ray.init(**ray_config)

def format_time(seconds):
    hh = int(seconds // 3600)
    mm = int(seconds % 3600 // 60)
    ss = int(seconds % 3600 % 60)
    return f'{hh:02}h {mm:02}m {ss:02}s'

def RemCocoWrapper(rem_coco):
    """Filters the relevant problems and uses tqdm to track progress.."""
    return tqdm(rem_coco.problems,
                leave=None,
                desc=f'<exp: {rem_coco.experiment_name}> Evaluating problems ({rem_coco.dim}D) on {rem_coco.host}',
                unit='problem',
                colour=rem_coco.bar_color)


class FitnessWrapper:
    """Wrapper to make a coco problem work with DiBB FitnessEvaluators."""

    def __init__(self, problem_id, observer_opts, dibb_id, run_number=1):
        self.problem_id = problem_id
        self.observer_opts = observer_opts
        self.dibb_id = dibb_id
        self.observer = None
        self.problem = None
        self.run_number = run_number
        cocoex.log_level(coco_config['log_level'])

    def _init_problem(self):
        hostname = platform.node()
        opts = self.observer_opts.format(prob=self.problem_id,
                                         host=hostname,
                                         block_id=self.block_id,
                                         run_number=self.run_number)
        self.observer = cocoex.Observer(coco_config['suite'], opts)
        self.problem = cocoex.Suite(coco_config['suite'], '', '').get_problem(self.problem_id, self.observer)

    def target_hit(self):
        if self.problem is None:
            return False
        return self.problem.final_target_hit

    def __call__(self, ind):
        if not self.problem:
            self._init_problem()
        return self.problem(ind)

    def __reduce__(self):
        """
        If problem is initialized, serialization will break the current observation
        and initialize the problem and observer again from scratch. Don't fetch
        the FitnessWrapper from the remote machine unless you are finished and you
        want to know the number of evaluations.
        """
        if self.problem:
            self.problem.free()
        return self.__class__, (self.problem_id, self.observer_opts, self.dibb_id, self.run_number)

    def __del__(self):
        if self.problem:
            self.problem.free()

# DiBB Hooks
def add_block_id(block_worker):
    if not hasattr(block_worker.fit_fn, 'block_id'):
        block_worker.fit_fn.block_id = block_worker.block_id


def check_target(block_worker):
    if block_worker.fit_fn.target_hit():
        block_worker.comm.set.remote('target_hit', val=True)
        block_worker.comm.set.remote('terminate', val=True)



@ray.remote
class RemCoco:
    """Ray actor for COCO"""

    def __init__(self, dibb_conf_init, dim, budget_multiplier, experiment_name, observer_opts, problem_indices=range(360), bar_color='#FFFFFF'):
        self.dibb = DiBB(**dibb_conf_init)
        self.dim = dim
        self.default_popsize = cma.CMAEvolutionStrategy([1] * self.dim, 1, {'verbose': -9}).popsize
        self.eval_budget = dim * budget_multiplier + 1
        self.evals_left = self.eval_budget
        self.experiment_name = experiment_name
        self.observer_opts = observer_opts
        self.bar_color = bar_color
        self.suite = cocoex.Suite(coco_config['suite'], '', f'dimensions: {self.dim}')
        self.problems = [self.suite[i] for i in problem_indices]
        self.host = platform.node()
        self.reset_config = {
            'optimizer_options': {
                'init_sigma': 2,
                'popsize': self.default_popsize,
                'verbose': -9,
            },
            'hooks': {
                'BlockWorker': [add_block_id, check_target]
            },
        }
        print('Initialized RemCoco instance on machine', self.host)

    def start(self):
        try:
            # Preparing run
            parent_dir = f'exdata/{self.experiment_name}'
            os.makedirs(parent_dir, exist_ok=True)

            print(f'Starting experiment {self.experiment_name}!')

            t0 = time.time()
            self.evals_left = self.eval_budget

            for i, problem in enumerate(RemCocoWrapper(self)):
                self.evals_left = self.eval_budget

                run_number = 0
                irestart = -1

                while self.evals_left > 0:
                    run_number += 1
                    irestart += 1

                    self.reset_config['fit_fn'] = FitnessWrapper(problem.id, self.observer_opts, self.dibb.dibb_id, run_number=run_number)
                    self.reset_config['init_ind'] = exp_config['init_ind_generator'](problem)

                    if exp_config['use_cma_ipop']:
                        # Will keep going at maximum popsize after 9 restarts in order to reach evaluation budget
                        popsize = 2 ** min(irestart, 9) * self.default_popsize 
                        self.reset_config['optimizer_options']['popsize'] = popsize

                    self.dibb.reset(**self.reset_config)
                    self.reset_target_hit()  # Add flag to communication dict again
                    self.dibb.optimize(nevals=self.evals_left)

                    self.evals_left -= sum(self.dibb.comm_dict['nevals'])

                    if self.target_hit():
                        break

                # Statistics and logging
                if self.evals_left < self.reset_config['optimizer_options']['popsize']:
                    reason = 'Evaluation budget reached'
                else:
                    assert self.target_hit(), 'Exited loop without proper termination condition!'
                    reason = 'Target hit'
                print(f'<exp: {self.experiment_name}> Finished evaluating {problem.id}',
                        f'<exp: {self.experiment_name}> Reason for termination: {reason}', sep='\n')

                nevals = self.dibb.comm_dict['nevals']
                print(f'> Eval stats for experiment {self.experiment_name}:', problem.id)
                print('>> evals done:', nevals, f'(Total: {sum(nevals)})')
                print('>> evals left:', self.evals_left)
                print('>> eval_budget:', self.eval_budget)
                print('>> nblocks:', self.dibb.nblocks)
                print('>> popsize:', self.dibb.optimizer_options['popsize'])
                print('>> ntrials_per_ind:', self.dibb.ntrials_per_ind)
                

            # Reset one last time to make observer write the last data
            self.dibb.reset(**self.reset_config)

            seconds_elapsed = time.time() - t0
            termination_msg = f'Experiment {self.experiment_name} finished after {format_time(seconds_elapsed)} ({int(seconds_elapsed)} seconds total)!'
            print(termination_msg)
            if exp_config['use_notify']:
                notify.send(termination_msg)
        except:
            print(traceback.format_exc())

    def target_hit(self):
        return ray.get(self.dibb.workers_comm.get.remote('target_hit'))

    def reset_target_hit(self):
        self.dibb.workers_comm.set.remote('target_hit', val=False)


# Format: {experiment_name: ObjRef}
experiments = {}
for i, exp in enumerate(exp_config['experiments']):
    exdata_folder = '_'.join(script_name.split('_')[:-1])
    observer_opts = 'algorithm_info: Distributed CMA-ES using DiBB  '
    observer_opts += 'algorithm_name: CMA-ES  '
    output_folder = f'{exdata_folder}/' + exp.name + '/{prob}_{host}_b{block_id:02}_r{run_number:02} '
    observer_opts += 'result_folder: ' + output_folder

    dibb_conf_init = {
        'nblocks': exp.nblocks,
        'ndims': exp.dimension,
        **dibb_config
    }

    experiments[exp.name] = RemCoco.options(resources={f'node:{exp.head_node_ip}': 1}).remote(
        dibb_conf_init=dibb_conf_init,
        dim=exp.dimension,
        budget_multiplier=coco_config['budget_multiplier'],
        experiment_name=exp.name,
        observer_opts=observer_opts,
        bar_color=exp.progress_bar_color,
        problem_indices=coco_config['problem_indices']
    )


experiments_obj_refs = {}

for experiment_name in experiments:
    if exp_config['use_notify']:
        notify.send(f'Starting experiment {experiment_name}!')
    experiments_obj_refs[experiment_name] = experiments[experiment_name].start.remote()

# Wait for all experiments, free up resources once an experiment terminates
while len(experiments_obj_refs) > 0:
    (ref, *_), _ = ray.wait(list(experiments_obj_refs.values()), num_returns=1)
    experiment_name = None
    for key, value in experiments_obj_refs.items():
        if value == ref:
            experiment_name = key
    del experiments_obj_refs[experiment_name]
    del experiments[experiment_name]

print(f'All experiments finished!')
