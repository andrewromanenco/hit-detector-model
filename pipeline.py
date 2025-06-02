import torch
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

PATCH_SIZE = 24

def hex_to_rgb(hex_color):
    hex_color = hex_color.strip("#")
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def load_model(model_path):
    sample_input = torch.randn(1, PATCH_SIZE, PATCH_SIZE)
    model = SimpleCNN(sample_input)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, PATCH_SIZE

def run_inference(
    model: torch.nn.Module,
    image: Image.Image,
    original: Image.Image,
    color: tuple,
    opacity: int,
    target_label: int,
    patch_size: int,
    stride: int = 4
):
    transform = transforms.ToTensor()

    width, height = image.size
    total_patches = ((width - patch_size) // stride + 1) * ((height - patch_size) // stride + 1)

    overlay = Image.new("RGBA", original.size, (0, 0, 0, 0))

    done = 0
    last_percent_reported = -1
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            tensor = transform(patch).unsqueeze(0)
            with torch.no_grad():
                pred = model(tensor)
                predicted_label = int(pred.item() > 0.9)

            if predicted_label == target_label:
                patch_overlay = Image.new("RGBA", (patch_size, patch_size), color + (opacity,))
                overlay.paste(patch_overlay, (x, y), patch_overlay)

            done += 1
            percent = int(done / total_patches * 100)
            if percent != last_percent_reported:
                print(f"\rProgress: {percent:3d}% ", end="", flush=True)
                last_percent_reported = percent

    print("\nDone.")
    blended = Image.alpha_composite(original.convert("RGBA"), overlay)
    return blended.convert("RGB")


class HitDetectorPipeline:
    def __init__(self, model_path="model.pt", color="#FF0000", opacity=128, target_label=1):
        self.model, self.patch_size = load_model(model_path)
        self.color = hex_to_rgb(color)
        self.opacity = opacity
        self.target_label = target_label

    def __call__(self, image: Image.Image) -> Image.Image:
        grayscale = image.convert("L")
        original = image.convert("RGB")
        return run_inference(
            model=self.model,
            image=grayscale,
            original=original,
            color=self.color,
            opacity=self.opacity,
            target_label=self.target_label,
            patch_size=self.patch_size
        )
