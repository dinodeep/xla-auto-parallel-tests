import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
os.environ["TF_CPP_MAX_LOG_LEVEL"] = "5"
# os.environ["TF_CPP_VMODULE"] = "cpu_compiler=5,gpu_compiler=5,cpu_client=5,sharding_propagation=5,auto_parallel=5"
os.environ["TF_CPP_VMODULE"] = (
    "cpu_compiler=5,"
    "auto_parallel=5,"
    "sharding_strategy_solver=5,"
    "complete_solver_builder=5,"
    "debug=5,"
    "sharding_strategy_selector=5,"
    "sharding_strategy_evaluator=5,"
    "sharding_enumeration=5,"
    "instruction_strategies=5,"
    "auto_sharding_strategy=5,"
    "auto_sharding=5,"
    "cpu_client=5"
)
# os.environ["TF_CPP_VMODULE"] = "auto_parallel=5,simple_solver_builder=5"

KB = 1000
MB = KB * KB
GB = MB * KB

replicated_flops_prop = 1
memory_limit_bytes = 1277952 // 8

# note: trailing space is necessary to enforce spacing for flags
os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=8 "
    "--xla_dump_to=/home/deepatel/dev/tests/dump "
    "--xla_auto_parallel_enable=true "
    f"--xla_auto_parallel_replicated_flops_prop={replicated_flops_prop} "
    f"--xla_auto_parallel_memory_limit_bytes={memory_limit_bytes} " 
)
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

print(os.environ["XLA_FLAGS"])