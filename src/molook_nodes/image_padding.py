import torch
import numpy as np


class ImageOutpaintPadding:
    """
    Pads an image to make its dimensions divisible by a specified value.

    This node takes an image and pads it to reach dimensions that are divisible by the specified value.
    The padding position and background color can be customized.

    Parameters:
    - image: Input image tensor
    - divisible_by: Value that the padded dimensions should be divisible by
    - position: Position of the original image within the padded canvas
    - color: Background color for padding (hex color value)

    Returns:
    - image: Padded image tensor
    - mask: Mask indicating the padding area (1 for padding, 0 for original image)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "divisible_by": (
                    "INT",
                    {"default": 8, "min": 1, "max": 256, "step": 1},
                ),
            },
            "optional": {
                "position": (
                    [
                        "top-left",
                        "top-center",
                        "top-right",
                        "right-center",
                        "bottom-right",
                        "bottom-center",
                        "bottom-left",
                        "left-center",
                        "center",
                    ],
                    {"default": "center"},
                ),
                "color": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFF,
                        "step": 1,
                        "display": "color",
                    },
                ),
            },
        }

    CATEGORY = "Molook_nodes/Image"

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")

    FUNCTION = "pad_image"

    def pad_image(
        self,
        image: torch.Tensor,
        divisible_by: int,
        position: str = "center",
        color: int = 0,
    ):
        """
        Pads an image to make its dimensions divisible by the specified value
        """
        # Get current dimensions (batch, height, width, channels)
        batch_size, orig_h, orig_w, channels = image.shape

        # Calculate required padding
        pad_h = (divisible_by - (orig_h % divisible_by)) % divisible_by
        pad_w = (divisible_by - (orig_w % divisible_by)) % divisible_by

        # If no padding needed, return original image and empty mask
        if pad_h == 0 and pad_w == 0:
            # Create an all-zeros mask (no padding)
            mask = torch.zeros((batch_size, 1, orig_h, orig_w), device=image.device)
            return (image, mask)

        # Calculate new dimensions
        new_h = orig_h + pad_h
        new_w = orig_w + pad_w

        # Convert hex color to RGB (0-1 range)
        color_r = ((color >> 16) & 0xFF) / 255.0
        color_g = ((color >> 8) & 0xFF) / 255.0
        color_b = (color & 0xFF) / 255.0

        # Create a new tensor filled with the specified color
        bg_color = torch.tensor(
            [color_r, color_g, color_b], dtype=torch.float32, device=image.device
        )
        padded_image = bg_color.view(1, 1, 1, 3).expand(
            batch_size, new_h, new_w, channels
        )

        # Create a mask to indicate padded areas (1 for padding, 0 for original image)
        mask = torch.ones((batch_size, new_h, new_w), device=image.device)

        # Calculate padding for each side based on position
        if position.startswith("top"):
            pad_top = 0
            pad_bottom = pad_h
        elif position.startswith("bottom"):
            pad_top = pad_h
            pad_bottom = 0
        else:  # center, left-center, right-center
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top

        if position.endswith("left"):
            pad_left = 0
            pad_right = pad_w
        elif position.endswith("right"):
            pad_left = pad_w
            pad_right = 0
        else:  # center, top-center, bottom-center
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

        # Special case for center position
        if position == "center":
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

        # Place the original image in the padded canvas
        padded_image[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w, :] = (
            image
        )
        mask[:, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w] = 0

        # Convert mask to expected format
        mask = mask.unsqueeze(1)

        return (padded_image, mask)


NODE_CLASS_MAPPINGS = {"ImageOutpaintPadding(Molook)": ImageOutpaintPadding}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {"ImageOutpaintPadding(Molook)": "Image Outpaint Padding"}
