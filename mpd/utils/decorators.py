import datetime
import os
import wandb
import yaml
from functools import wraps


def save_args(exp_dir, kwargs, filename='experiment_args.yml'):
    filtered = {}
    for key, value in kwargs.items():
        if type(value) is tuple or type(value) is int or type(value) is float or type(value) is bool or type(
                value) is str or value is None:
            filtered[key] = value
    with open(os.path.join(exp_dir, filename), 'w') as f:
        yaml.safe_dump(filtered, f)


def save_module_args(exp_dir, args, filename='module_args.yml'):
    save_args(exp_dir, args, filename=filename)


def load_args(exp_dir, filename='experiment_args.yml'):
    with open(os.path.join(exp_dir, filename), 'r') as f:
        args = yaml.safe_load(f)
    return args


def load_module_args(exp_dir, filename='module.yml'):
    return load_args(exp_dir, filename=filename)


def update_args(exp_dir, partial_args):
    args = load_args(exp_dir)
    for key, value in partial_args.items():
        args[key] = value
    save_args(exp_dir, args)


def evaluation(eval_func):
    @wraps(eval_func)
    def wrapper(**kwargs):
        experiment_args = load_args(kwargs["exp_dir"])
        # Run the experiment
        eval_func(experiment_args, **kwargs)

    return wrapper


def filter_kwargs(kwargs, blacklist=['device', 'exp_dir']):
    model_kwargs = {}
    for key, value in kwargs.items():
        if key not in blacklist and not key.endswith('_field'):
            model_kwargs[key] = value
    return model_kwargs


def pretrain_helper(model_load_function):
    """
    Saves relevant model kwargs to a yml file (default is module.yml).
    """

    @wraps(model_load_function)
    def wrapper(**kwargs):

        model_kwargs = filter_kwargs(kwargs)

        # Inject submodels if any
        submodule_kwargs = None
        if "submodules" in kwargs:
            submodule_kwargs = {}
            for module_name, submodule in kwargs["submodules"].items():
                kwargs[module_name] = submodule
                submodule_kwargs[module_name] = filter_kwargs(submodule._all_kwargs)
            model_kwargs['submodules'] = submodule_kwargs

        save_module_args(kwargs['exp_dir'], model_kwargs)

        # Run the experiment
        model = model_load_function(**kwargs)

        setattr(model, '_all_kwargs', kwargs)

        return model

    return wrapper


def model_loader(model_load_function):
    @wraps(model_load_function)
    def wrapper(**kwargs):
        # Inject submodels if any
        if "submodules" in kwargs:
            for module_name, submodule in kwargs["submodules"].items():
                kwargs[module_name] = submodule

        # Run the experiment
        model = model_load_function(**kwargs)

        # Save submodules in a dictionary (for saving, ...)
        model.submodules = kwargs["submodules"] if "submodules" in kwargs else {}

        return model

    return wrapper
"""

def single_experiment(exp_func):
    @wraps(exp_func)
    def wrapper(*args, **kwargs):
        # Make results directory
        assert 'results_dir' in kwargs and 'seed' in kwargs, "results_dir and seed must be arguments"
        results_dir = os.path.join(kwargs['results_dir'], str(kwargs['seed']))
        os.makedirs(results_dir, exist_ok=True)
        kwargs['results_dir'] = results_dir

        # Save arguments
        save_args(results_dir, kwargs)

        # Fix seed
        fix_random_seed(kwargs['seed'])

        # WandB
        if kwargs['wandb_silent']:
            os.environ["WANDB_SILENT"] = "true"

        init = {"project": kwargs['project'],
                "reinit": True,
                "entity": kwargs['entity'],
                "notes": datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                "config": kwargs}

        if 'group' in kwargs and kwargs["group"] is not None:
            init["group"] = kwargs['group']
        if 'tags' in kwargs and kwargs["tags"] is not None:
            init["tags"] = kwargs['tags']
        if 'run_name' in kwargs and kwargs["run_name"] is not None:
            init["name"] = kwargs['run_name']

        run = wandb.init(**init)

        # Run the experiment
        exp_func(*args, **kwargs)

        run.finish()

    return wrapper


def experiment(exp_func):
    @wraps(exp_func)
    def wrapper(opt):
        # Make results directory
        root_dir = opt.saving_root
        opt.exp_dir = str(os.path.join(root_dir, opt.experiment_name, str(opt.seed)))

        exists = os.path.exists(os.path.join(opt.exp_dir, 'checkpoints')) or \
                 os.path.exists(os.path.join(opt.exp_dir, 'summaries'))

        if exists:
            exit('Experiment already exists.')
        os.makedirs(opt.exp_dir, exist_ok=True)

        # Save arguments
        save_args(opt.exp_dir, opt)

        # Run the experiment
        exp_func(opt)

    return wrapper"""
