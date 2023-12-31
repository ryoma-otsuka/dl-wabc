import os
import subprocess
from logging import getLogger

logger = getLogger(__name__)

DEFAULT_ATTRIBUTES = (
    "index",
    "uuid",
    "name",
    "timestamp",
    "memory.total",
    "memory.free",
    "memory.used",
    "utilization.gpu",
    "utilization.memory",
)


def get_gpu_status(gpu_id, keys=DEFAULT_ATTRIBUTES):
    """Get detail status of selected GPU.
    Args:
        gpu_id (str): GPU ID (e.g. 0, GPU-fa0ee1b3-7d00-deca-12b3-32b8cf608096)
            If you want to get status of several GPUs, use CSV style notation,
            (e.g. 0,1,)
        keys (tuple of str): query key.
    Returns:
        list of GPU status dicts.
    """
    cmd = "nvidia-smi --id={} --query-gpu={} --format=csv,noheader".format(
        gpu_id, ",".join(keys)
    )
    output = subprocess.check_output(cmd, shell=True)
    logger.debug("Get GPU status:")
    logger.debug("... cmd    = {}".format(cmd))
    logger.debug("... output = {} ".format(output))
    lines = output.decode().split("\n")
    lines = [line.strip() for line in lines if line.strip() != ""]
    return [{k: v for k, v in zip(keys, line.split(", "))} for line in lines]


def get_available_devices(multi_gpu=False):
    """Returns available GPU device keys (e.g. cuda:0).
    Args:
        multi_gpu: Boolean, If True, set up  multi GPUs.
    Return:
        str: device id
    Note:
        This functions assumes Single GPU for Single Process. When multiple
        GPUs are found, this function returns the first GPU.
    """
    logger.info(
        "$NVIDIA_VISIBLE_DEVICE = {}".format(
            os.environ.get("NVIDIA_VISIBLE_DEVICES", None)
        )
    )
    logger.info(
        "$CUDA_VISIBLE_DEVICE = {}".format(
            os.environ.get(
                "CUDA_VISIBLE_DEVICES",
                None)))

    # If $NVIDIA_VISIBLE_DEVICES is enmpy, check $CUDA_VISIBLE_DEVICES
    if "NVIDIA_VISIBLE_DEVICES" in os.environ.keys():
        visible_devices = os.environ["NVIDIA_VISIBLE_DEVICES"].split(",")
    elif "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        visible_devices = []
    logger.info("Found CUDA devices = {}".format(visible_devices))

    if len(visible_devices) == 0:
        logger.warning("No GPUs are available. Use cpu instead.")
        return "cpu"

    if multi_gpu:
        raise NotImplementedError("Multi-GPU setting has not supported yet.")
    else:
        gpu_status = get_gpu_status(",".join(visible_devices))
        device_id = "cuda:{}".format(gpu_status[0]["index"])
        logger.info(
            "Selected Device = {} (uuid={})".format(
                device_id,
                gpu_status[0]["uuid"]))
    return device_id
