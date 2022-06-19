import numpy as np
import glob
import os
import re
import shutil
from datetime import timedelta
from tqdm import tqdm

# Merging without FitnessEvaluators
def get_problem_directories(base_dir, problem):
    """Returns all directories containing the string `problem`"""
    dirs = [d for d in glob.glob(os.path.join(base_dir, '*')) if problem in d]
    # Consistency check
    nmachines = len(set([re.findall(r'diufrm\d{3}', d.split('/')[-1]).pop(0) for d in dirs]))
    nblocks = len(set([re.findall(r'b\d{2}', d.split('/')[-1]).pop(0) for d in dirs]))
    assert nmachines == nblocks, 'Error! Number of machines != number of blocks'
    return dirs

def group_by_runs(directories):
    """Group directories by runs"""
    run_numbers = sorted(list(set([re.findall(r'r\d{2}', d)[0] for d in directories])))
    runs = [[d for d in directories if run_nb in d] for run_nb in run_numbers]
    return runs

def get_data_paths(directory_path, file_extension=None):
    """Returns one or more paths to the data files corresponding to a specific problem directory."""
    if file_extension:
        assert file_extension in ['tdat', 'rdat', 'dat', 'info'], 'File extension must either be None (which yields all data files) or one of `tdat`, `rdat`, `dat` or `info`.'
    paths = list(glob.glob(os.path.join(directory_path, 'data_f*', '*.*dat')))
    paths.append(list(glob.glob(os.path.join(directory_path, '*.info')))[0])
    if file_extension:
        try:
            return [p for p in paths if p.split('.')[-1] == file_extension][0]
        except:
            raise Exception(f'There seems to be no data file with extension {file_extension}' +
                             'in the specified directory!')
    return paths

def get_repaired_info(path, suite_name='bbob'):
    """Opens a (corrupted) coco *.info file, repairs it and returns it as a string."""
    with open(path, 'r', errors='replace') as file:
        text = file.read().split('funcId')
        text = f"suite = '{suite_name}', funcId" + text[-1]
        return text


def is_monotonic(sequence, mode='decreasing', strict=True):
    assert mode in ['increasing', 'decreasing']
    if len(sequence) < 2:
        return True

    def monotony_satisfied(prev, succ):
        if strict and mode == 'increasing':
            return succ > prev
        elif strict and mode == 'decreasing':
            return succ < prev
        elif mode    == 'increasing':
            return succ >= prev
        else:
            return succ <= prev

    for idx in range(1, len(sequence)):
        prev = sequence[idx-1]
        succ = sequence[idx]
        if not monotony_satisfied(prev, succ):
            return False
    return True

def is_integer(num):
    return np.floor(num) == num

def merge(base_dir, problem_id, out_dir='merged', suite_name='bbob'):
    import time
    t0 = time.time()

    # Scan directories
    probs = get_problem_directories(base_dir, problem_id)

    # Create groups of different runs
    runs = group_by_runs(probs)

    for current_run in runs:
        for file_extension in ['dat', 'tdat']:
            # Keep track of best fitness score so far (actually it is fitness - Fopt
            # -> the difference between the fitness score and the optimum)
            best_fit = float('inf')

            data_paths = [get_data_paths(p, file_extension) for p in current_run]

            # Open the data files and save handles into list
            files = [open(path, 'r', encoding='utf-8', errors='replace') for path in data_paths]

            # Check if the headers are the same (they should be)
            headers = set()
            for f in files:
                headers.add(f.readline())
            if len(headers) > 1:
                raise Exception('There is at least one file with a different header! ' +
                                'Are you merging the correct files?')

            # Read data of all tdat or dat files in this run
            lines_per_file = [file.readlines() for file in files]
            data = [[line.split(' ') for line in lines] for lines in lines_per_file]
            for file in files:
                file.close()

            # Keep track of where are in each file
            line_pointers = [0] * len(data)

            header = headers.pop()
            out_file = [header]
            original_eval_index = 0
            merged_eval_index = 0

            # Loop until there is no data in the list anymore
            while data:
                original_eval_index += 1

                # Get current line's data from each of the data files
                data_pool = [data[file_idx][file_pointer] for file_idx, file_pointer in enumerate(line_pointers)]
                evals = [int(line_data[0]) for line_data in data_pool]
                current_min_eval = min(evals)

                if current_min_eval == original_eval_index:
                    # At this point, we know that we have to look at the different lines
                    current_eval_data_indices = [idx for idx, e in enumerate(evals) if e == current_min_eval]
                    current_eval_data = [data_pool[idx] for idx in current_eval_data_indices]

                    # Sort indices in increasing fitness score order
                    # This is better than using fits.index because it considers also duplicates (unique indices)
                    fits = [float(d[2]) for d in current_eval_data]
                    sorted_fit_indices = [t[1] for t in sorted(list(zip(fits, range(len(fits)))), reverse=True)]

                    # Number of evals that weren't logged
                    num_unlogged = len(data_pool) - len(current_eval_data)
                    merged_eval_index += num_unlogged

                    for idx in sorted_fit_indices:
                        merged_eval_index += 1
                        fit = fits[idx]
                        if fit < best_fit or file_extension == 'tdat':
                            # The fitness has increased, save line for writing it later
                            best_fit = min(fit, best_fit)
                            line = current_eval_data[idx].copy()

                            # Adapt evaluation number to reflect the merged one
                            line[0] = str(merged_eval_index)
                            out_file.append(' '.join(line))

                        # Move pointer in corresponding file
                        # assert data_pool.count(before_removing[idx]) == 1, "Two identical rows!"
                        data_pool_file_index = current_eval_data_indices[idx]
                        line_pointers[data_pool_file_index] += 1

                        # The file has ended, set removal flag in data and line_pointers
                        if line_pointers[data_pool_file_index] >= len(data[data_pool_file_index]):
                            data[data_pool_file_index] = None

                    # Remove files/columns that were marked for deletion
                    line_pointers = [p for idx, p in enumerate(line_pointers) if data[idx]]
                    data = [d for d in data if d]
                else:
                    # At this point, we know that the min_eval is larger than the counter
                    # Thus we can just add `len(data_pool)` evals to the merging_counter
                    # (these were unlogged/skipped evals in the original data)
                    merged_eval_index += len(data_pool)

            # Write dat/tdat files
            content = ''.join(out_file)
            run_number = runs.index(current_run) + 1  # Starts always at 1
            exp_folder = os.path.basename(base_dir)
            subdir_and_fname = '/'.join(data_paths[0].split('/')[-2:])
            # import ipdb; ipdb.set_trace()
            out_path = os.path.join(out_dir, exp_folder, problem_id + f'_run_{run_number:02}', subdir_and_fname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w') as file:
                file.write(content)

        # Write (copy) rdat file
        rdat_path = get_data_paths(current_run[0], 'rdat')
        with open(rdat_path, 'r') as file:
            content = file.read()
        out_path = os.path.join(os.path.dirname(out_path), os.path.basename(rdat_path))
        with open(out_path, 'w') as file:
            file.write(content)

        # Write info file
        info_path = get_data_paths(current_run[0], 'info')
        fname = os.path.basename(info_path)
        info_content = get_repaired_info(info_path, suite_name=suite_name).split(':')[0]
        formatted_best_fit = f'{best_fit:.1e}'
        info_content += f':{merged_eval_index}|{formatted_best_fit}'
        out_path = os.path.join(os.path.dirname(os.path.dirname(out_path)), fname)
        with open(out_path, 'w') as file:
            file.write(info_content)


def postprocess(experiment_dir):
    paths = glob.glob(f'{experiment_dir}/*')
    problem_ids = list(set([os.path.basename(p).split('_run')[0] for p in paths]))
    problem_ids = sorted(problem_ids)
    runs_per_problem = [sorted([p for p in paths if problem_id in p]) for problem_id in problem_ids]

    # Tdat checkpoints that are logged for a budget multiplier up to 2e7 on largest problem (640d)
    # Was calculated as follows: int(10 ** (203 / 20)) / 640 = 2e7
    tdat_eval_checkpoints = [int(10 ** (i / 20)) for i in range(203)]

    for runs in tqdm(runs_per_problem):
        if len(runs) == 1:
            # Remove the _run suffix from the directory
            os.rename(runs[0], runs[0].split('_run')[0])
            continue  # No need to merge, there is only one run!

        offsets = [0]
        for run in runs[:-1]:
            info_path = get_data_paths(run, 'info')
            with open(info_path, 'r') as file:
                last_eval = int(re.findall(r':(\d+)\|', file.readlines()[-1])[0])
                offsets.append(last_eval)

        # Keep track of best fitness score across dat and tdat files
        best_fit = float('inf')

        # if runs[0] == '/media/luca/Evo-980/coco/merged/210813_bbob_thesis/40d_8b/bbob_f010_i01_d20_run_01':
        #     breakpoint()

        # Trim the data files
        for file_ext in ['dat', 'tdat']:
            output_lines = []

            # Variables used for trimming *.dat lines according to the formulas
            # that are used in the logging of a sequential observer
            dat_target_formula_exponent = 999  # Exponent i in 10^(i/5)

            # Concatenate data
            for run_idx, run in enumerate(runs):
                file_path = get_data_paths(run, file_ext)
                with open(file_path, 'r') as file:
                    # Skip header if it is not the first run
                    if run_idx != 0:
                        file.readline()
                    for line in file.readlines():
                        output_lines.append(line)

            # Postprocess and trim data
            last_eval_num = 0
            run_number = 1
            for line_idx in range(1, len(output_lines)):
                parts = output_lines[line_idx].split(' ')
                eval_num = int(parts[0])
                fit = float(parts[2])

                if eval_num < last_eval_num:
                    # We encountered another run
                    run_number += 1
                last_eval_num = eval_num

                # BEFORE: if file_ext == 'dat' and fit >= best_fit:
                if file_ext == 'dat' and fit >= 10 ** (dat_target_formula_exponent / 5):
                    # Mark line for deletion
                    output_lines[line_idx] = None
                    continue
                elif file_ext == 'tdat' and eval_num in tdat_eval_checkpoints:
                    # If eval num was not computed using the forumla, it was logged
                    # due to a block worker that met a stopping criterion (-> paused)
                    # This should go into the tdat file. Otherwise, check if the
                    # new evaluation number after the concatenation is still compliant
                    # with the tdat checkpoint formula
                    offset = sum(offsets[:run_number])
                    new_eval_num = eval_num + offset
                    if new_eval_num not in tdat_eval_checkpoints:
                        # Mark line for deletion
                        output_lines[line_idx] = None
                        continue

                if fit < best_fit:
                    best_fit = fit

                if file_ext == 'dat':
                    # We encountered a fitness score that is smaller than 10^(i/5)
                    # Thus, we compute the new i
                    quantity = 5 * np.log10(fit)
                    if is_integer(quantity):
                        dat_target_formula_exponent = quantity + 1
                    else:
                        dat_target_formula_exponent = np.ceil(quantity)

                # Modify evaluation number and save back in list
                offset = sum(offsets[:run_number])
                parts[0] = str(eval_num + offset)
                output_lines[line_idx] = ' '.join(parts)

            # Remove lines that were marked for deletion
            output_lines = [line for line in output_lines if line]

            # Consistency check
            if file_ext == 'dat':
                evals = [int(line.split(' ')[0]) for line in output_lines[1:]]
                assert is_monotonic(evals, mode='increasing',
                                    strict=True), 'Evaluation numbers are not strictly increasing!'

            # Update `last_eval_num` with last non-deleted eval number
            last_eval_num = int(output_lines[-1].split(' ')[0])

            file_path_to_overwrite = get_data_paths(runs[0], file_ext)
            content = ''.join(output_lines)
            with open(file_path_to_overwrite, 'w') as file:
                file.write(content)

        # Update info file with `last_eval_num` and `best_fit`
        info_path = get_data_paths(runs[0], 'info')
        with open(info_path, 'r') as file:
            content = file.readlines()
            last_line = content[-1].split(':')[0]
            last_line += f':{last_eval_num}|{best_fit:.1e}'
            content[-1] = last_line
            content = ''.join(content)

        # Overwrite file with merged data info
        with open(info_path, 'w') as file:
            file.write(content)

        # Delete merged directories
        for run in runs[1:]:
            shutil.rmtree(run)

        # Remove the _run suffix from the directory
        os.rename(runs[0], runs[0].split('_run')[0])


def merge_experiment_data(experiment_directory, out_dir, suite_name='bbob'):
    experiment_name = os.path.basename(experiment_directory)

    prob_ids = sorted(
        list(set(os.path.basename('_'.join(p.split('_')[:-3]))
                 for p in glob.glob(os.path.join(experiment_directory, '*')))))
    for prob_id in tqdm(prob_ids):
        merge(experiment_directory, prob_id, out_dir=out_dir, suite_name=suite_name)

    postprocess(f'{out_dir}/{experiment_name}')

def get_all_instances(fn_id, directories, sort=True):
    out = []
    tmp_problem_id = os.path.basename(directories[0])
    suite, _, instance, dim = tmp_problem_id.split('_')
    fn_num = f'f{fn_id:03}'
    for directory in directories:
        folder_name = os.path.basename(directory)
        s, f, i, d = folder_name.split('_')
        if s == suite and f == fn_num and d == dim:
            out.append(directory)
    return sorted(out) if sort else out

def merge_instances(fn_num, directories):
    # Get problem instance directories
    instances = get_all_instances(fn_num, directories)

    # Merge info files across instances
    info_path = glob.glob(os.path.join(instances[0], '*.info'))[0]
    with open(info_path, 'a') as info:
        for instance in instances[1:]:
            instance_info_path = glob.glob(os.path.join(instance, '*.info'))[0]
            with open(instance_info_path, 'r') as instance_info:
                last_line = instance_info.readlines()[-1]
                summary = last_line.split('.dat')[-1]
                info.write(summary)

    # Merge data files across instances
    for file_ext in ['dat', 'rdat', 'tdat']:
        data_path = glob.glob(os.path.join(instances[0], f'data_f*/*.{file_ext}'))[0]
        with open(data_path, 'a') as data:
            for instance in instances[1:]:
                instance_data_path = glob.glob(os.path.join(instance, f'data_f*/*.{file_ext}'))[0]
                with open(instance_data_path, 'r') as instance_data:
                    content = instance_data.read()
                    data.write(content)

    # Copy content to parent directory
    parent_dir = os.path.dirname(instances[0])
    for source in os.listdir(instances[0]):
        shutil.move(os.path.join(instances[0], source), parent_dir)

    # Delete instance folders
    for instance in instances:
        shutil.rmtree(instance)

if __name__ == '__main__':
    out_dir = '/insert/output/path/here'
    experiments = glob.glob('/insert/path/to/experiment/data/here/*')

    for experiment in experiments:
        merge_experiment_data(experiment, out_dir, suite_name='bbob') # or bbob-largescale

    for experiment in glob.glob(os.path.join(out_dir, '*')):
        problem_dirs = glob.glob(os.path.join(experiment, '*'))

        # Only merge the instances that were used during the experiment
        for i in range(1, 6):
            merge_instances(i, problem_dirs)
        for i in range(10, 15):
            merge_instances(i, problem_dirs)
