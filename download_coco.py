import fiftyone
import fiftyone.zoo as foz

# List available zoo datasets
print(foz.list_zoo_datasets())

# Download the COCO-2017 validation split and load it into FiftyOne
dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    label_types=["detections", "segmentations"],
    split="train",
    max_samples=30000,
)
dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    label_types=["detections", "segmentations"],
    split="validation",
    max_samples=3000,
)
# dataset = fiftyone.zoo.load_zoo_dataset(
#     "coco-2017",
#     label_types=["detections", "segmentations"],
#     split="test",
#     max_samples=1000,
# )

# # Visualize it in the App
# session = fiftyone.launch_app(dataset)
# session.wait()

