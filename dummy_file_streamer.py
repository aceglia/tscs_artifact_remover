import numpy as np

from biosiglive.streaming.async_client import AsyncTCPClient
from artifact_remover.io_utils import DataLoader
import asyncio
from biosiglive.streaming.utils import CircularBuffer, RollingBuffer

async def main(data, chunk_size=20):
    client = AsyncTCPClient('127.0.0.1', 12345)
    await client.connect()
    chunks = [
        data[:, i:i+chunk_size]
        for i in range(0, data.shape[-1], chunk_size)
    ]
    dt_sample = 1 / 2000
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

def test_buffer_main(data, chunk_size=20):
    chunks = [
        data[:, i:i+chunk_size]
        for i in range(0, data.shape[-1], chunk_size)
    ]
    dt_sample = 1 / 2000
    t0 = 0.0
    count = 0
    import time
    t_next = time.perf_counter()
    circ_buffer = CircularBuffer(data.shape[0], 1000, dt=dt_sample)
    roll_buffer = RollingBuffer(data.shape[0], 1000)
    dt = 0.01
    while True:
        t_next += dt
        if count >= data.shape[-1]:
            count = 0
        data_tmp = chunks[count % len(chunks)]
        t = t0 + np.arange(chunk_size) * dt_sample
        t0 += chunk_size * dt_sample

        circ_buffer.append(data_tmp, t)
        roll_buffer.append(data_tmp, t)

        now = time.perf_counter()
        sleep_time = t_next - now


        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            t_next = now
        if count == 3000:
            print("circ_buffer time:", circ_buffer.get_time()[-5:])
            print("roll_buffer time:", roll_buffer.get_time()[-5:])
        count += 1
import time
n_iter = 20000
def benchmark(buffer, name):
    data_stream = np.random.rand(10, 20)
    time_stream = np.arange(20) * 0.001
    if name == "RollingBuffer":
        buffer = RollingBuffer(10, 10000)
    if name == "CircularBuffer":
        buffer = CircularBuffer(10, 10000, dt=0.001)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        buffer.append(data_stream, time_stream)
    t1 = time.perf_counter()

    # include read cost
    t_read0 = time.perf_counter()
    for _ in range(n_iter):
        _, _ = buffer.get()
    t_read1 = time.perf_counter()

    print(f"\n{name}")
    print(f"Append time: {(t1 - t0) / n_iter:.4f} s")
    print(f"Read time:   {(t_read1 - t_read0) / n_iter:.4f} s")

if __name__ == "__main__":
    
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r"D:\Downloads\T1_008_arm_sa_002.mat"
    # path_file = r"test001.txt"
    # path_file = r'D:\Documents\Programmation\tscs_artifact_remover\test_data\007Loc_sa_20_Avec000.mat'
    path_file =  r"D:\Documents\Programmation\tscs_artifact_remover\test_data\clean_walk_testwith_artifacts_synth_80_steps.bio"
    loader = DataLoader(path_file, center=True)
    loader._apply_stack_batch()
    # test_buffer_main(loader.init_data, chunk_size=20)
    asyncio.run(main(loader.init_data, chunk_size=20))
    # benchmark(CircularBuffer(10, 5000, dt=0.001), "CircularBuffer")
    # benchmark(RollingBuffer(10, 5000), "RollingBuffer")