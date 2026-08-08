import os
from torch.utils.data import Dataset
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = list(os.listdir(root))
        self.files = [p for p in self.files if p.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root, self.files[idx])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load corrupted image {image_path}: {e}. Falling back to another image.")
            next_idx = (idx + 1) % len(self.files)
            image_path = os.path.join(self.root, self.files[next_idx])
            image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

        
def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)



def adaptive_instance_normalization(content_feat, style_feat):
    # input is [batch_size, channels, height, width]
    # adaIn usues the mean and std of the style features to normalize the content features,so we calculated the mean and std of the style features and then we normalized the content features using the mean and std of the style features
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)#
    content_mean, content_std = calc_mean_std(content_feat) # from these the AdaIN gets the mean and std
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size) # here we are normalizing the content features using the mean and std of the content features
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size) 
    


def calc_mean_std(feat, eps=1e-5): # here we reshaping the output size as same as the input size i.e [batch_size, channels, 1, 1]
    # [batch_size, channels, h, w]
    size = feat.size()
    assert(len(size)==4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1)
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std
