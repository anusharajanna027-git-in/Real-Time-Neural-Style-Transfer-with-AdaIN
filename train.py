import argparse
import torch
from pathlib import Path
from utils.model import Decoder, VGGEncoder
from utils.utils import ImageFolderDataset, get_transform, adaptive_instance_normalization, calc_mean_std
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--content_dir', type=str, default='./content_data')
    parser.add_argument('--style_dir', type=str, default='./style_data')
    parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')
    parser.add_argument('--experiment', type=str, default='experiment1', help='Name of experiment')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--final_size', type=int, default=256, help='Size of the final Image')
    parser.add_argument('--content_size', type=int, default=512, help='Size of the content image')
    parser.add_argument('--style_size', type=int, default=512, help='Size of the style image')
    parser.add_argument('--crop', action='store_true', default=True, help='Crop the image')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=5e-5, help='Learning rate decay')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--content_weight', type=float, default=1.0, help='Weight for content loss')
    parser.add_argument('--style_weight', type=float, default=1.0, help='Weight for style loss')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging')
    parser.add_argument('--save_interval', type=int, default=2, help='Interval for saving model checkpoints')
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training from checkpoint')
    parser.add_argument('--decoder_path', type=str, default=None, help='Path to the decoder checkpoint')
    parser.add_argument('--optimizer_path', type=str, default=None, help='Path to the optimizer checkpoint')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum training steps per epoch')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA requested or defaulted, but PyTorch cannot detect a CUDA-capable GPU. Falling back to CPU.")
        print("To enable GPU training, install PyTorch with CUDA support (e.g. cu118 / cu121).")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    save_dir = Path('experiment') / args.experiment
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save argument values
    with open(save_dir / 'args.txt', 'w') as args_file:
        for key, value in vars(args).items():
            args_file.write(f'{key}: {value}\n')

    content_transform = get_transform(args.content_size, args.crop, args.final_size)
    style_transform = get_transform(args.style_size, args.crop, args.final_size)

    content_dataset = ImageFolderDataset(args.content_dir, content_transform)
    style_dataset = ImageFolderDataset(args.style_dir, style_transform)

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    style_dataloader = DataLoader(style_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    print('Number of batches in content dataset:', len(content_dataloader))
    print('Number of batches in style dataset:', len(style_dataloader))

    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)

    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: 1.0 / (1.0 + args.lr_decay * epoch)
    )

    if args.resume:
        if args.decoder_path and Path(args.decoder_path).exists():
            decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
            print(f"Resumed decoder weights from {args.decoder_path}")
        if args.optimizer_path and Path(args.optimizer_path).exists():
            optimizer.load_state_dict(torch.load(args.optimizer_path, map_location=device))
            print(f"Resumed optimizer state from {args.optimizer_path}")

    mse_loss = torch.nn.MSELoss()
    encoder.eval()

    total_batches = min(len(content_dataloader), len(style_dataloader))
    if args.max_steps:
        total_batches = min(total_batches, args.max_steps)

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_closs = 0.0
        running_sloss = 0.0
        step = 0

        progress_bar = tqdm(zip(content_dataloader, style_dataloader), total=total_batches)
        for content_batch, style_batch in progress_bar:
            step += 1
            if args.max_steps and step > args.max_steps:
                break
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)

            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)

            t = adaptive_instance_normalization(c_feats[-1], s_feats[-1])

            g = decoder(t)

            g_feats = encoder(g)

            loss_c = mse_loss(g_feats[-1], t) * args.content_weight

            loss_s = 0.0
            for g_f, s_f in zip(g_feats, s_feats):
                g_mean, g_std = calc_mean_std(g_f)
                s_mean, s_std = calc_mean_std(s_f)
                loss_s += mse_loss(g_mean, s_mean) + mse_loss(g_std, s_std)

            loss_s = loss_s * args.style_weight

            loss = loss_c + loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"Epoch [{epoch+1}/{args.epochs}] Loss:{loss.item():.4f}, Content Loss:{loss_c.item():.4f}, Style Loss:{loss_s.item():.4f}")

            running_loss += loss.item()
            running_closs += loss_c.item()
            running_sloss += loss_s.item()

        scheduler.step()

        if total_batches > 0:
            running_loss /= total_batches
            running_closs /= total_batches
            running_sloss /= total_batches

        print(f"Epoch [{epoch + 1}/{args.epochs}], Avg Loss: {running_loss:.4f}, Content Loss: {running_closs:.4f}, Style Loss: {running_sloss:.4f}")

        if (epoch + 1) % args.log_interval == 0:
            torch.save(decoder.state_dict(), save_dir / f'decoder_{epoch + 1}.pth')
            torch.save(decoder.state_dict(), 'decoder.pth')
            torch.save(optimizer.state_dict(), save_dir / f'optimizer_{epoch + 1}.pth')
            print(f"Saved decoder checkpoint to {save_dir / f'decoder_{epoch + 1}.pth'} and root decoder.pth")

            with torch.no_grad():
                output = torch.cat([content_batch, style_batch, g], dim=0)
                save_image(output, save_dir / f'output_{epoch + 1}.png', nrow=args.batch_size)

if __name__ == "__main__":
    main()