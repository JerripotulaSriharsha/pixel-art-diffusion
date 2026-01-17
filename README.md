# AI Pixel Art Generation

The goal of this project is to train a stable diffusion model to generate enemy pixel sprites in the style of the Dragon Quest games.

I create a youtube video to explain the project in detail: [AI Pixel Art Generation](https://youtu.be/OdafuDeNMyM)

## Example Model Output

<img src="data/sample_outputs/jelly_monster.png" alt="Jelly Monster" width="256"/>

Prompt: `jelly, monster, dq_pookie`

For more examples, check `data/sample_outputs/`.


## Project Structure

Main directories:

- `data`: Contains raw sprite sheets and cleaned sprites ready for training.
- `inference`: Contains the script for generating new sprites.
- `training`: Contains the scripts for training the model.
- `preprocessing`: Contains the scripts for preprocessing the dataset.
- `postprocessing`: Contains the scripts for cleaning model outputs.

## Dataset Structure

The dataset is organized into the following directories:

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/pixel-art-generation.git
```

Navigate to the project directory:

```bash
cd pixel-art-generation
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1) if you haven't already.

Install dependencies:

```bash
uv sync
```

## Usage

### Inference

To comfortably generate sprites you will need around 16GB of VRAM.

Adapt the `inference.py` script to your needs and run it.

```bash
uv run inference/inference.py
```

### Training

To comfortably train the model you will need around 24GB of VRAM. (I trained on A100 40GB)

Go to the experiments directory:

```bash
cd training/experiments
```

Create a shell script adapted to your needs and run it.

For example:

```bash
./lora_v1.sh
```

## Collaborations

Do you have cool AI projects you want to collaborate on? Let me know!
