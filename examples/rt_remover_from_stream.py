import numpy as np

from star_emg.rt_automatic_remover import RtArtifactRemover
from star_emg.solution import Solution

if __name__ == "__main__":
    path_file = r"data\test001.txt"
    artifact_remover = RtArtifactRemover(window_size=500, data=path_file, signal_filter=False, center=True, chunk_size=20, data_rate=2000)
    is_data = True
    global_output = None
    global_intput = None
    while is_data:
        is_data, chunk = artifact_remover.streamer.get_next_chunk(artifact_remover.streamer.chunk_size)
        if not is_data:
            break
        output = artifact_remover.process_chunk(chunk[0:1], hankel_size=150, freq_bounds=[10, 200], factor=0.2)
        if output is None:
            continue
        global_output = output if global_output is None else np.concatenate((global_output, output), axis=-1)
        global_intput = chunk[0:1] if global_intput is None else np.concatenate((global_intput, chunk[0:1]), axis=-1)

    sol_dict = {
        "init_data": global_intput,
        "output": global_output,
    }
    sol = Solution(data_rate=artifact_remover.streamer.data_rate)
    sol.from_dict(sol_dict)
    sol.analyse()
    sol.plot(signals=True, fft=True, singular_values=False, stack_batch=False, show_analysis=True)

