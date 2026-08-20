<div align="center">

# 🎨 StyleForge AI
### Real-Time Arbitrary Neural Style Transfer with Adaptive Instance Normalization (AdaIN)

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](http://makeapullrequest.com)

<p align="center">
  <b>Transform any photo into a masterpiece in real-time using deep neural networks and custom style blending.</b>
</p>

[Key Features](#-key-features) • [Demo Results](#-demo-results) • [How It Works](#-how-it-works) • [Quick Start](#-quick-start) • [Project Structure](#-project-structure) • [Training](#-training-your-own-decoder)

---

</div>

## ✨ Key Features

- ⚡ **Real-Time Feed-Forward Inference**: Instantaneous style synthesis without the hundreds of iterative optimization steps required by traditional NST.
- 🎭 **Arbitrary Style Transfer**: Apply **any** style image (paintings, sketches, abstract textures) onto any content image without needing to retrain the network.
- 🎚️ **Interactive Style Control ($\alpha$)**: Adjust stylization strength seamlessly from original content ($\alpha=0.0$) to intense artistic rendering ($\alpha=1.0$).
- 🌐 **Modern Web Application**: Sleek dark-mode interface built with Flask, responsive design, drag-and-drop uploads, and instant result downloads.
- 💻 **Cross-Platform & Hardware Agnostic**: Runs on **Windows**, **macOS**, and **Linux**, automatically detecting and leveraging CUDA GPUs or falling back to CPU.

---

## 🖼️ Demo Results

| Content Image | Style Reference | Stylized Output |
| :---: | :---: | :---: |
| <img src="Demo_IO_Images/i-p/i_p%20image.jpg" width="220" alt="Content"/> | <img src="Demo_IO_Images/i-p/style%201.png" width="220" alt="Style 1"/> | <img src="Demo_IO_Images/o-p/o_p%20style%201.jpg" width="220" alt="Output 1"/> |
| <img src="Demo_IO_Images/i-p/i_p%20image.jpg" width="220" alt="Content"/> | <img src="Demo_IO_Images/i-p/style%202.jpg" width="220" alt="Style 2"/> | <img src="Demo_IO_Images/o-p/o_p%20style%202.jpg" width="220" alt="Output 2"/> |

---

## 🧠 How It Works

This project is an implementation of **Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization** (*Huang & Belongie, ICCV 2017*).

```
 ┌─────────────────┐      ┌───────────────┐      ┌─────────────────────────┐
 │  Content Image  │ ───► │  VGG Encoder  │ ───► │      Content Feats      │
 └─────────────────┘      └───────────────┘      └────────────┬────────────┘
                                                              │
                                                              ▼
                                                 ┌─────────────────────────┐
                                                 │       AdaIN Layer       │ ──► [Alpha Blend] ──► ┌───────────────────┐ ──► ┌──────────────────┐
                                                 │ (Mean & Std Alignment)  │                       │  Trained Decoder  │     │  Stylized Image  │
                                                 └────────────▲────────────┘                       └───────────────────┘     └──────────────────┘
 ┌─────────────────┐      ┌───────────────┐                   │
 │   Style Image   │ ───► │  VGG Encoder  │ ──────────────────┘
 └─────────────────┘      └───────────────┘
```

### 1. Feature Encoding (VGG-19)
The fixed encoder (VGG-19 sliced at layer `relu4_1`) extracts multi-scale feature maps from both the content $f(c)$ and style $f(s)$ images.

### 2. Adaptive Instance Normalization (AdaIN)
The AdaIN layer shifts and scales the normalized content activations to match the spatial mean ($\mu$) and standard deviation ($\sigma$) of the style activations:

$$\text{AdaIN}(c, s) = \sigma(s) \left( \frac{f(c) - \mu(c)}{\sigma(c)} \right) + \mu(s)$$

### 3. Style Interpolation ($\alpha$)
The feature maps are blended according to the user-selected $\alpha \in [0, 1]$ parameter:

$$t = \alpha \cdot \text{AdaIN}(c, s) + (1 - \alpha) \cdot f(c)$$

### 4. Convolutional Decoder
The learnable mirror Decoder network inverts the feature tensor $t$ back into a reconstructed RGB image.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/StyleForge-AI.git
cd StyleForge-AI
```

### 2. Set Up Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Web Application
```bash
cd NST_Code
python app.py
```
Open **`http://localhost:5000`** in your browser.

---

## 📁 Project Structure

```text
ai-nst-project/
├── Demo_IO_Images/            # Sample inputs and stylized demo outputs
│   ├── i-p/                   # Input sample images
│   └── o-p/                   # Generated output samples
├── NST_Code/
│   ├── experiment/            # Checkpoints & training sample outputs
│   │   └── final_exp/
│   │       └── decoder_final.pth   # Pre-trained Decoder model
│   ├── static/                # Static web assets & dynamic uploads
│   ├── templates/
│   │   └── index.html         # Modern web application UI
│   ├── utils/
│   │   ├── models.py          # PyTorch VGGEncoder & Decoder architectures
│   │   └── utils.py           # AdaIN algorithm & dataset transforms
│   ├── app.py                 # Flask web server & inference controller
│   ├── train.py               # Decoder training pipeline
│   └── vgg_normalised.pth     # Pre-trained normalized VGG-19 weights
├── .gitignore                 # Excluded environments and caches
├── Procfile.txt               # Deployment definition for cloud hosting
├── README.md                  # Project documentation
└── requirements.txt           # Python package dependencies
```

---

## 🏋️ Training Your Own Decoder

To train the decoder on your custom dataset (e.g., MS-COCO for content and WikiArt for style):

```bash
python train.py \
  --content_dir /path/to/content_dataset \
  --style_dir /path/to/style_dataset \
  --vgg vgg_normalised.pth \
  --experiment my_custom_experiment \
  --epochs 20 \
  --batch_size 8 \
  --lr 1e-4 \
  --content_weight 1.0 \
  --style_weight 5.0
```

---

## 🛠️ Tech Stack

- **Deep Learning**: PyTorch, Torchvision
- **Web Backend**: Flask, Flask-WTF, WTForms, Werkzeug, Gunicorn
- **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), Bootstrap 5, FontAwesome
- **Image Processing**: Pillow (PIL), NumPy

---

## 📚 References & Acknowledgments

- **AdaIN Paper**: *Huang, X., & Belongie, S. (2017). "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization." ICCV 2017.* [[Paper]](https://arxiv.org/abs/1703.06868)
- **VGG Network**: *Simonyan, K., & Zisserman, A. (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition."* [[Paper]](https://arxiv.org/abs/1409.1556)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.