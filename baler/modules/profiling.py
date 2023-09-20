from torch.profiler import profile, record_function, ProfilerActivity
from codecarbon import EmissionsTracker
import cProfile
import io
import pstats
import torch
from pstats import SortKey


def energy_profiling(project_name, measure_power_secs):
    """Energy Profiling measure the amount of electricity that 
    was consumed by decorated function f and amount of CO(2) emission.
    It utilize the codecarbon package for tracking this information.
    Args:
        f (_type_): decorated function
    """
    def decorator(f):
        """_summary_

        Args:
            f (_type_): _description_
        """
        def inner_function(*args, **kwargs):
            """_summary_

            Returns:
                _type_: _description_
            """
            tracker = EmissionsTracker(
<<<<<<< HEAD
                project_name=project_name,
=======
                project_name=project_name, 
>>>>>>> 3d671de (Added profilers and electricity meters)
                measure_power_secs=measure_power_secs)
            #
            tracker.start_task(f'f.__name__')
            try:
                val = f(*args, **kwargs)
            finally:
                emissions = tracker.stop_task()
                print("CO2 emission [kg]: ", emissions.emissions)
<<<<<<< HEAD
                print("CO2 emission rate [kg/h]: ", 3600*emissions.emissions_rate)
=======
>>>>>>> 3d671de (Added profilers and electricity meters)
                print("CPU energy consumed [kWh]: ", emissions.cpu_energy)
                print("GPU energy consumed [kWh]: ", emissions.gpu_energy)
                print("RAM energy consumed [kWh]: ", emissions.ram_energy)
            return val
        return inner_function
    return decorator


def pytorch_profile(f):
    """This function perform the pytorch profiling of CPU, GPU time and memory 'consumed'
    of the function f execution 

    Args:
        f (_type_): decorated function
    """
    def inner_function(*args, **kwargs):
        """_summary_
<<<<<<< HEAD

        Returns:
            _type_: _description_
=======
>>>>>>> 3d671de (Added profilers and electricity meters)
        """
        if torch.cuda.is_available():
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        else:
            activities = [ProfilerActivity.CPU]
        #Start profiler before the function will be executed
        with profile(activities=activities,
<<<<<<< HEAD
                     on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./logs/{f.__name__}", worker_name='worker0'),
                     schedule=torch.profiler.schedule(
                         wait=0, warmup=0, active=1, repeat=1),
                     record_shapes=True,
                     with_stack=True,
                     profile_memory=True) as prof:
            with record_function(f"{f.__name__}"):
=======
                     on_trace_ready=torch.profiler.tensorboard_trace_handler('baler', worker_name='worker0'),
                     schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1),
                     record_shapes=True,
                     with_stack=True,
                     profile_memory=True) as prof:
            with record_function(f'f.__name__'):
>>>>>>> 3d671de (Added profilers and electricity meters)
                #Call the decorated function
                val = f(*args, **kwargs)
                prof.step()
                prof.stop()
<<<<<<< HEAD
        print(prof.key_averages().table(sort_by="cpu_time_total"))
        #Store the information about CPU and GPU usage
        if torch.cuda.is_available():
            prof.export_stacks("profiler_stacks.json","self_cuda_time_total")
        #Store the results to the .json file
        prof.export_stacks("profiler_stacks.json", "self_cpu_time_total")
=======
        #Store the information about CPU and GPU usage
        if torch.cuda.is_available():
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            print(prof.key_averages().table(sort_by="cuda_memory_usage"))
            prof.export_stacks("/tmp/profiler_stacks.txt",
                               "self_cuda_time_total")
        print(prof.key_averages().table(sort_by="cpu_memory_usage"))
        print(prof.key_averages().table(sort_by="cpu_time_total"))
        #Store the results to the .json file
        prof.export_stacks("/tmp/profiler_stacks.txt", "cpu_time_total")
>>>>>>> 3d671de (Added profilers and electricity meters)
        return val
    return inner_function


def c_profile(func):
    """_summary_

    Args:
        func (_type_): _description_
    """
    def wrapper(*args, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE  # 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return wrapper
<<<<<<< HEAD
=======


>>>>>>> 3d671de (Added profilers and electricity meters)
