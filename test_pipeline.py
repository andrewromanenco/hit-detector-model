from PIL import Image
from pipeline import HitDetectorPipeline

pipe = HitDetectorPipeline("model.pt")

img = Image.open("input.png")
result = pipe(img)
result.save("output.png")
print("✅ Output saved to output.png")
