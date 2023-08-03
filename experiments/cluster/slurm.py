import os
from os import path
import subprocess
from collections.abc import Iterable

dirname = os.path.dirname(os.path.realpath(__file__))
template_file = f"{dirname}/slurm-template.sh"
JOB_NAME = "${JOB_NAME}"
PARTITION_OPTION = "${PARTITION_OPTION}"
MEMORY_OPTION = "${MEMORY_OPTION}"
EXCLUSIVE = "${EXCLUSIVE}"
COMMAND_PLACEHOLDER = "${COMMAND_PLACEHOLDER}"
GIVEN_NODE = "${GIVEN_NODE}"
LOAD_ENV = "${LOAD_ENV}"
WORK_DIR = "${WORK_DIR}"
ARRAY_SIZE = "${ARRAY_SIZE}"
DEPENDENCY = "${DEPENDENCY}"
array_template_file = path.join(dirname, "array_template.sh")
work_dir = "/p/projects/ou/labs/gane/satisfia/stable-baselines3-contrib-satisfia/experiments/"


def value_to_arg(value):
    if isinstance(value, Iterable) and not isinstance(value, str):
        return " ".join(map(str, value))
    elif isinstance(value, bool):
        assert value, "Boolean argument must be True"
        # If False, we should not add the --arg, so to support bool=False, we would need to check for bool args in the caller
        return ""
    else:
        return str(value)


def dict_to_str_args(args):
    return " ".join([f"--{k} {value_to_arg(v)}" for k, v in args.items()])


def submit_job_array(
    python_file,
    args: dict,
    n_jobs,
    experiment_name,
    post_python_file=None,
    post_args: dict = None,
    testing=False,
    wandb_sync=False,
):
    """
    Submit a job array to the cluster.
    A job array is a job that will run n_jobs times the same command. Each worker will be able to access its index in the array with `os.environ["SLURM_ARRAY_TASK_ID"]`.
    The job array will be named `experiment_name` and the logs will be saved in {work_dir}/logs/slurm/{experiment_name.out}.
    If post_python_file is not None, a job will be submitted after the array job is finished. This job will have a dependency on the array job.
    This function will create a script file in ./logs/array_{experiment_name}.sh and submit it to the cluster if testing=False. Else, it will only create the script file.
    :param python_file: The python file to run. It should be in the same directory as this file.
    :param args: A dictionary of arguments to pass to the python file. The arguments will be passed as --arg_key arg_value.
    :param n_jobs: The number of jobs in the array.
    :param experiment_name: The name of the job array.
    :param post_python_file: The python file to run after the array job is finished. It should be in the same directory as this file. If None, no post job will be submitted.
    :param post_args: A dictionary of arguments to pass to the post python file. The arguments will be passed as --arg_key arg_value.
    :param testing: If True, the script file will be created but not submitted to the cluster.
    :param wandb_sync: If True, wandb will be synced before the post job is submitted. Check https://wandb.ai/ for more information.


    """
    # Assert python file and post_python file exist
    python_file = path.join(dirname, python_file)
    if post_python_file is not None:
        post_python_file = path.join(dirname, post_python_file)
    assert os.path.isfile(python_file), f"File {python_file} does not exist"
    if post_python_file is not None:
        assert os.path.isfile(post_python_file), f"File {post_python_file} does not exist"
    with open(array_template_file, "r") as f:
        text = f.read()
    text = (
        text.replace(COMMAND_PLACEHOLDER, f"python3 {python_file} {dict_to_str_args(args)}")
        .replace(ARRAY_SIZE, str(n_jobs - 1))
        .replace(DEPENDENCY, "")
        .replace(JOB_NAME, experiment_name)
        .replace(WORK_DIR, work_dir)
    )
    script_file = path.join(dirname, "logs", f"array_{experiment_name}.sh")
    with open(script_file, "w") as f:
        f.write(text)
    if not testing:
        output = subprocess.run(["sbatch", "--parsable", script_file], capture_output=True, text=True)
        print(f"Job {output} submitted! Script file is at: {script_file}. Log file is at: ./logs/slurm/{experiment_name}.out")
    else:
        output = None
    if post_python_file:
        if post_args is None:
            post_args = {}
        # Add a job with a dependency on the array job
        if not testing:
            job_id = output.stdout.strip()
            job_id = int(job_id if job_id.isdigit() else job_id.split(";")[0])
            print(f"Post job will be dependent on job {job_id}")
        else:
            job_id = "{DUMMY_JOB_ID}"
        with open(array_template_file, "r") as f:
            text = f.read()
        text = (
            text.replace(COMMAND_PLACEHOLDER, f"python3 {post_python_file} {dict_to_str_args(post_args)}")
            .replace(ARRAY_SIZE, "0") # We only want one job
            .replace(DEPENDENCY, f"#SBATCH --dependency=afterok:{job_id}")
            .replace(JOB_NAME, f"post_{experiment_name}")
            .replace(WORK_DIR, work_dir)
        )
        script_file = path.join(dirname, "logs", f"post_{experiment_name}.sh")
        with open(script_file, "w") as f:
            f.write(text)
        if not testing:
            subprocess.run(["sbatch", script_file])
        if wandb_sync:
            if args.get("log_path") is None:
                raise ValueError("log_path must be specified in args to sync with wandb")
            if not testing:
                subprocess.run(
                    [
                        "sbatch",
                        "--partition=io",
                        "--qos=io",
                        "-D",
                        work_dir,
                        f"--output=./logs/slurm/{experiment_name}wandb.out",
                        f"--error=./logs/slurm/{experiment_name}wandb.err",
                        "-J",
                        f"wandb_{experiment_name}",
                        "--dependency",
                        f"afterok:{job_id}",
                        "--wrap",
                        f'"wandb sync {args["log_path"]}"',
                    ]
                )
