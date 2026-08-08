# Real-Time Neural Style Transfer with AdaIN

A real-time neural style transfer web application built with **PyTorch** and **Flask**, implementing the **Adaptive Instance Normalization (AdaIN)** method from the paper [*Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization*](https://arxiv.org/abs/1703.06868) by Huang & Belongie (2017).

---

## 🎨 Demo

Upload any content image and style image, adjust the **style strength (alpha)** slider, and get a stylized output in seconds — powered by your trained AdaIN decoder running on GPU or CPU.

---

## 🧠 How It Works

AdaIN transfers the style of an image by aligning the **mean and standard deviation** of the content feature maps to match those of the style image in the VGG feature space:

```
AdaIN(x, y) = σ(y) * ((x - μ(x)) / σ(x)) + μ(y)
```

- **Encoder**: Pre-trained VGG-19 (frozen) extracts deep feature representations.
- **AdaIN Layer**: Aligns content features to style statistics.
- **Decoder**: A trained lightweight CNN reconstructs the stylized image from AdaIN features.

---

## 🗂️ Project Structure

```
├── app.py                  # Flask web application
├── train.py                # Training script
├── utils/
│   ├── model.py            # VGGEncoder and Decoder model definitions
│   └── utils.py            # Dataset, transforms, AdaIN, loss utilities
├── templates/
│   └── index.html          # Web UI (dark-mode, glassmorphism design)
├── static/
│   └── uploads/            # User-uploaded and stylized images
├── examples/               # Example output images shown on the web page
├── vgg_normalised.pth      # Pre-trained VGG encoder weights
├── decoder.pth             # Trained decoder weights (after training)
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/anusharajanna027-git-in/Real-Time-Neural-Style-Transfer-with-AdaIN.git
cd Real-Time-Neural-Style-Transfer-with-AdaIN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install PyTorch with CUDA (Recommended for GPU training)

```bash
# For Python 3.14 with CUDA 12.6 (nightly)
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126

# For Python 3.12/3.13 with CUDA 12.1 (stable)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Download VGG Weights

Download `vgg_normalised.pth` and place it in the project root:

```bash
# Direct download (recommended)
curl -o vgg_normalised.pth https://www.hal.t.u-tokyo.ac.jp/~inoue/projects/tmp/vgg_normalised.pth
```

Or manually download from:
- **[vgg_normalised.pth](https://www.hal.t.u-tokyo.ac.jp/~inoue/projects/tmp/vgg_normalised.pth)** (~80MB, hosted by the PyTorch AdaIN author Naoto Inoue)

---

## 🏋️ Training

Prepare your content and style image datasets, then run:

```bash
python train.py \
  --batch_size 4 \
  --epochs 2 \
  --experiment "my_experiment" \
  --content_dir "path/to/content_data" \
  --style_dir "path/to/style_data" \
  --device cuda
```

### Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--batch_size` | 1 | Batch size |
| `--epochs` | 2 | Number of training epochs |
| `--lr` | 1e-4 | Learning rate |
| `--content_weight` | 1.0 | Content loss weight |
| `--style_weight` | 1.0 | Style loss weight |
| `--device` | auto | `cuda` or `cpu` |
| `--experiment` | experiment1 | Name of output folder |

### Resume Training from Checkpoint

```bash
python train.py \
  --epochs 5 \
  --experiment "my_experiment" \
  --content_dir "path/to/content_data" \
  --style_dir "path/to/style_data" \
  --device cuda \
  --resume \
  --decoder_path "experiment/my_experiment/decoder_2.pth" \
  --optimizer_path "experiment/my_experiment/optimizer_2.pth"
```

---

## 🌐 Running the Web App

```bash
python app.py
```

Then open your browser at **http://localhost:5000**

### Features:
- Upload any content + style image (JPG/PNG)
- Adjustable **alpha slider** (0.0 = content only, 1.0 = full style)
- Download stylized output
- Examples section showing real model outputs

---

## 📊 Training Details

| Setting | Value |
|---|---|
| Dataset | WikiArt (style) + COCO/custom (content) |
| Epochs | 2 |
| Batch Size | 4 |
| Image Size | 256×256 (final), 512×512 (resize) |
| GPU | NVIDIA GeForce RTX 3050 Laptop |
| Optimizer | Adam (lr=1e-4) |

---

## 📦 Requirements

```
torch
torchvision
flask
flask-wtf
flask-bootstrap
wtforms
pillow
tqdm
```

---

## 📄 References

- Huang, X., & Belongie, S. (2017). [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868). ICCV 2017.
- Johnson, J. et al. (2016). [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

---

## 👤 Author

**Anusha Rajanna** — [@anusharajanna027-git-in](https://github.com/anusharajanna027-git-in)