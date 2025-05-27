import textwrap
from typing import Optional

from PIL import Image, ImageDraw, ImageFont


def render_text_on_image(
    language_instruction: str, width: int, height: int
) -> Optional[Image.Image]:
    font_size = 55
    try:
        text_image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(text_image)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        max_width = 2000
        lines = textwrap.wrap(
            language_instruction, width=int(max_width / (font_size / 2))
        )
        line_height = 20
        total_text_height = len(lines) * line_height
        y = (height - total_text_height) // 2

        for line in lines:
            draw.text((50, y), line, fill="black", font=font)
            y += line_height

        return text_image
    except Exception as e:
        print(f"Error rendering text as image: {e}")
        return None
