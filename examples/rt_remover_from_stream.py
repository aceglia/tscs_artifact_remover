from star_emg.rt_automatic_remover import RtArtifactRemover
from star_emg.streaming_utils import DataStreamer

if __name__ == "__main__":
    path_file = r"data\test001.txt"
    streamer = DataStreamer(path_file, chunk_size=20)
    artifact_remover = RtArtifactRemover(window_size=500)
    is_data = True
    while is_data:
        is_data, chunk = streamer.get_next_chunk(artifact_remover.streamer.chunk_size)
        if not is_data:
            break
        output = artifact_remover.process_chunck(chunk, hankel_size=150, freq_bounds=[10, 200], factor=0.2)
