import numpy as np

from biosiglive.streaming.async_client import AsyncTCPClient
from star_emg.io_utils import DataLoader
import asyncio
from biosiglive.streaming.utils import CircularBuffer, RollingBuffer

async def main(data, chunk_size=20, data_rate=2000):
    client = AsyncTCPClient('127.0.0.1', 12345)
    await client.connect()
    chunks = [
        data[:, i:i+chunk_size]
        for i in range(0, data.shape[-1], chunk_size)
    ]
    
    dt_sample = 1 / data_rate
    t0 = 0.0
    count = 0
    import time
    t_next = time.perf_counter()
    dt = 0.01
    while True:
        t_next += dt
        if count >= data.shape[-1]:
            count = 0
        data_tmp = chunks[count % len(chunks)]
        t = np.round(t0 + np.arange(chunk_size) * dt_sample, 4)
        t0 += np.round(chunk_size * dt_sample, 4)
        await client.send_array(data_tmp, sample_time=t)

        now = time.perf_counter()
        sleep_time = t_next - now

        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
        else:
            t_next = now
        count += 1


if __name__ == "__main__":
    
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r"D:\Downloads\T1_008_arm_sa_002.mat"
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\test_data\test001.txt"
    # path_file = r"test001.txt"
    # path_file = r'D:\Documents\Programmation\tscs_artifact_remover\test_data\007Loc_sa_20_Avec000.mat'
    path_file =  r"D:\Documents\Programmation\tscs_artifact_remover\test_data\clean_walk_testwith_artifacts_synth_50_steps.bio"
    path_file =  r"D:\Documents\Programmation\tscs_artifact_remover\test_data\test001with_artifacts_synth_50_steps.bio"

    # path_file =  r"D:\Documents\Programmation\tscs_artifact_remover\test_data\005TMSgaitpdt_001.mat"

    loader = DataLoader(path_file, center=True)
    loader._apply_stack_batch()
    # test_buffer_main(loader.init_data, chunk_size=20)
    asyncio.run(main(loader.init_data, chunk_size=20, data_rate=loader.data_rate))
    # benchmark(CircularBuffer(10, 5000, dt=0.001), "CircularBuffer")
    # benchmark(RollingBuffer(10, 5000), "RollingBuffer")