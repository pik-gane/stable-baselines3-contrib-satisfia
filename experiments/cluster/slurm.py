import argparse
import os
from os import path
import subprocess
import time
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


# def submit_job(exp_name, memory=4, node=None, processor="cpu", duration="short", load_env="", command="", exclusive=False):
#
#     I only use submit_job_array, but I keep this function for reference. It's not up to date.
#
#     if node:
#         node_info = "#SBATCH -w {}".format(node)
#     else:
#         node_info = ""
#
#     job_name = "{}_{}".format(exp_name, time.strftime("%m%d-%H%M", time.localtime()))
#
#     memory_option = f"#SBATCH --mem={memory * 1_000}" if memory else ""
#
#     qos = "gpu" + duration if processor == "gpu" else duration
#     partition = "gpu" if processor == "gpu" else "standard"
#     partition_option = f"""
# #SBATCH --partition={partition}
# #SBATCH --qos={qos}
# {"#SBATCH --gres=gpu:k40m:1" if processor == "gpu" else ""}
#     """
#
#     exclusive = "#SBATCH --exclusive" if exclusive else ""
#
#     # ===== Modified the template script =====
#     with open(template_file, "r") as f:
#         text = f.read()
#     text = (
#         text.replace(JOB_NAME, job_name)
#         .replace(PARTITION_OPTION, partition_option)
#         .replace(MEMORY_OPTION, memory_option)
#         .replace(EXCLUSIVE, exclusive)
#         .replace(COMMAND_PLACEHOLDER, command)
#         .replace(LOAD_ENV, str(load_env))
#         .replace(GIVEN_NODE, node_info)
#     )
#
#     # ===== Save the script =====
#     script_file = os.path.join(dirname, "logs/{}.sh".format(job_name))
#     with open(script_file, "w") as f:
#         f.write(text)
#
#     # ===== Submit the job =====
#     subprocess.Popen(["sbatch", script_file], env=os.environ)
#     print(
#         "Job submitted! Script file is at: {}. Log file is at: {}".format(script_file, "./slurm/logs/{}.out".format(job_name))
#     )
#

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
    script_file = f"./logs/array_{experiment_name}.sh"
    with open(script_file, "w") as f:
        f.write(text)
    if not testing:
        output = subprocess.run(["sbatch", "--parsable", script_file], capture_output=True, text=True)
        print(f"Job {output} submitted! Script file is at: {script_file}. Log file is at: ./slurm/logs/{experiment_name}.out")
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
            .replace(ARRAY_SIZE, str(0))
            .replace(DEPENDENCY, f"#SBATCH --dependency=afterok:{job_id}")
            .replace(JOB_NAME, f"post_{experiment_name}")
            .replace(WORK_DIR, work_dir)
        )
        script_file = f"./logs/post_{experiment_name}.sh"
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
                        "--background",
                        f"--output=./slurm/logs/{experiment_name}wandb.out",
                        f"--error=./slurm/logs/{experiment_name}wandb.err",
                        "-J",
                        f"wandb_{experiment_name}",
                        "--dependency",
                        f"afterok:{job_id}",
                        "--wrap",
                        f'"wandb sync {args["log_path"]}"',
                    ]
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        required=True,
        help="The job name and path to logging file (exp_name.log).",
    )
    parser.add_argument("--memory", "-m", type=int, help="The memory to use (in GB).", default=4)
    parser.add_argument(
        "--node",
        "-w",
        type=str,
        help="The specified nodes to use. Same format as the " "return of 'sinfo'. Default: ''.",
    )
    parser.add_argument(
        "--processor",
        "-p",
        type=str,
        choices=["gpu", "cpu"],
        default="cpu",
        help="Whether to use a gpu or a cpu",
    )
    parser.add_argument(
        "--duration",
        "-d",
        choices=["short", "medium", "long"],
        type=str,
        default="short",
        help="slurm qos duration (short, medium, long)",
    )
    parser.add_argument(
        "--load-env",
        type=str,
        help="The script to load your environment ('module load cuda/10.1')",
        default="",
    )
    parser.add_argument(
        "--command",
        type=str,
        required=True,
        help="The command you wish to execute. For example: "
        " --command 'python test.py'. "
        "Note that the command must be a string.",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--exclusive", action="store_true", default=False, help="Whether to reserve a whole node exclusively for this job."
    )

    args = parser.parse_args()

    if args.node:
        node_info = "#SBATCH -w {}".format(args.node)
    else:
        node_info = ""

    job_name = "{}_{}".format(args.exp_name, time.strftime("%m%d-%H%M", time.localtime()))

    memory_option = f"#SBATCH --mem={args.memory * 1_000}" if args.memory else ""

    qos = "gpu" + args.duration if args.processor == "gpu" else args.duration
    partition = "gpu" if args.processor == "gpu" else "standard"
    partition_option = f"""
#SBATCH --partition={partition}
#SBATCH --qos={qos}
{"#SBATCH --gres=gpu:k40m:1" if args.processor == "gpu" else ""}
    """

    exclusive = "#SBATCH --exclusive" if args.exclusive else ""

    command = args.command

    # ===== Modified the template script =====
    with open(template_file, "r") as f:
        text = f.read()
    text = (
        text.replace(JOB_NAME, job_name)
        .replace(PARTITION_OPTION, partition_option)
        .replace(MEMORY_OPTION, memory_option)
        .replace(EXCLUSIVE, exclusive)
        .replace(COMMAND_PLACEHOLDER, command)
        .replace(LOAD_ENV, str(args.load_env))
        .replace(GIVEN_NODE, node_info)
    )

    # ===== Save the script =====
    script_file = os.path.join(dirname, "logs/{}.sh".format(job_name))
    with open(script_file, "w") as f:
        f.write(text)

    # ===== Submit the job =====
    print("Starting to submit job!")
    subprocess.Popen(["sbatch", script_file], env=os.environ)
    print(
        "Job submitted! Script file is at: {}. Log file is at: {}".format(script_file, "./slurm/logs/{}.out".format(job_name))
    )
