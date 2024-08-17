from tqdm.auto import tqdm, trange  # type: ignore
import jax_dataloader as jdl  # type: ignore

def loader_loop(loader: jdl.DataLoader, desc: str):
    pbar = tqdm(loader, desc=desc, leave=False)
    for step, (orig_img, true_seg_img) in enumerate(pbar):
        # free_mem, total_mem = torch.cuda.mem_get_info()
        # get gpu usage

        # used_mem_mb = (total_mem - free_mem) / 1024**2
        # total_mem_mb = total_mem / 1024**2
        # gpu_util = torch.cuda.utilization()

        # live_stats = OrderedDict()
        # live_stats["GPU util"] = f"{gpu_util:.2f}"
        # live_stats["GPU Memory util"] = f"{used_mem_mb:.2f}MB/{total_mem_mb:.2f}MB"
        # pbar.set_postfix(live_stats)

        yield pbar, step, (orig_img, true_seg_img)
