from biosiglive.streaming.async_client import AsyncTCPClient
from artifact_remover.io_utils import DataLoader
import asyncio

async def main(data, chunck_size=20):
    client = AsyncTCPClient('127.0.0.1', 12345)
    await client.connect()

    count = 0
    while count < data.shape[-1]:
        data_tmp = data[:, count:count+chunck_size]
        await client.send_array(data_tmp)
        await asyncio.sleep(0.01)  # ~100 Hz
        count += 1

if __name__ == "__main__":
    path_file = r"D:\Documents\Programmation\tscs_artifact_remover\004_TN-SCI_002.txt"
    path_file = r"D:\Downloads\T1_008_arm_sa_002.mat"
    # path_file = r"test001.txt"
    # path_file = r'D:\Documents\Programmation\tscs_artifact_remover\test_data\007Loc_sa_20_Avec000.mat'
    # path_file =  r"D:\Documents\Programmation\tscs_artifact_remover\test_data\clean_walk_testwith_artifacts_synth_80_steps.bio"
    loader = DataLoader(path_file, center=True)
    loader._apply_stack_batch()
    asyncio.run(main(loader.init_data, chunck_size=20))