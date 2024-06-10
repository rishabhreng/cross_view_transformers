import hydra
from pathlib import Path
from omegaconf import OmegaConf
import torch
import numpy as np
import imageio
from cross_view_transformer.common import setup_experiment, load_backbone

@hydra.main(
    config_path='/home/vrb230004/cross_view_transformers/config/', 
    config_name='config.yaml')
def main(cfg):
    # resolve config references
    OmegaConf.resolve(cfg)

    print(list(cfg.keys()))

    # Additional splits can be added to cross_view_transformer/data/splits/nuscenes/
    SPLIT = 'val_qualitative_000'
    SUBSAMPLE = 5

    model, data, viz = setup_experiment(cfg)

    dataset = data.get_split(SPLIT, loader=False)
    dataset = torch.utils.data.ConcatDataset(dataset)
    dataset = torch.utils.data.Subset(dataset, range(0, len(dataset), SUBSAMPLE))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    print(len(dataset))


    
    # Download a pretrained model (13 Mb)
    # MODEL_URL = 'https://www.cs.utexas.edu/~bzhou/cvt/cvt_nuscenes_vehicles_50k.ckpt'
    # CHECKPOINT_PATH = '../logs/cvt_nuscenes_vehicles_50k.ckpt'
    # CHECKPOINT_PATH = '/home/vrb230004/cross_view_transformers/logs/cross_view_transformers_test/0606_233625/checkpoints/model.ckpt'
    CHECKPOINT_PATH = '/home/vrb230004/cross_view_transformers/logs/cvt_nuscenes_vehicles_50k.ckpt'
    if Path(CHECKPOINT_PATH).exists():
        network = load_backbone(CHECKPOINT_PATH)
    else:
        network = model.backbone

        print(f'{CHECKPOINT_PATH} not found. Using randomly initialized weights.')

    GIF_PATH = '/home/vrb230004/cross_view_transformers/scripts/predictions2.gif'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    network.to(device)
    network.eval()

    images = list()

    with torch.inference_mode():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = network(batch)
            # print(type(pred))
            # print(pred['bev'].shape)
            # print(pred['bev'])

            visualization = np.vstack(viz(batch=batch, pred=pred))

            images.append(visualization)


    # Save a gif
    # duration = [2 for _ in images[:-1]] + [100 for _ in images[-1:]]
    duration = 600
    imageio.mimsave(GIF_PATH, images, duration=duration)
    

if __name__ == '__main__':
    main()
