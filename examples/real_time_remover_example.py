from artifact_remover.rt_automatic_remover import RtArtifactRemover
from artifact_remover.streaming_utils import DataStreamer

if __name__ == "__main__":
    path_file = r"data\test001.txt"
    streamer = DataStreamer(path_file, chunk_size=20)
    artifact_remover = RtArtifactRemover(window_size=500)
    is_data = True
    while is_data:
        is_data, chunk = streamer.get_next_chunk(artifact_remover.streamer.chunk_size)
        if not is_data:
            break
        artifact_remover.process_chunck(chunk, hankel_size=150, freq_bounds=[10, 200], factor=0.2)

    # result_dict = {
    #     "output": out.flatten(),
    #     "unfiltered_signal": out.flatten(),
    #     "data": all_data.flatten(),
    # }
    # sol = Solution(data_rate=artifact_remover.streamer.data_loader.data_rate)
    # sol.from_notch_filter(result_dict, (out.shape[0], 1, process_window))
    # sol.plot(signals=True, fft=True, singular_values=False, stack_batch=False, show_analysis=False)
