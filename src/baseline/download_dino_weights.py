from transformers import AutoModel, AutoImageProcessor

AutoImageProcessor.from_pretrained("facebook/dinov2-small").save_pretrained("./dinov2-small")
AutoModel.from_pretrained("facebook/dinov2-small").save_pretrained("./dinov2-small")