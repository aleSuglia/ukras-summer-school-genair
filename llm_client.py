import io
import re
from typing import Optional

from ai2thor.server import Event
from openai import OpenAI
from PIL import Image

from responses import AgentActionResponse, AgentResponse, AgentTextualResponse


class LLMClient:
    """
    Client for interacting with a language model. Handles communication with the model
    and processes responses.

    Attributes:
        model_name (str): The name of the language model to use.
        history (list): A history of interactions with the model.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        system_prompt=None,
    ) -> None:
        self.model_name = model_name
        self.history = []
        assert system_prompt is not None, "System prompt cannot be None."
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def reset(self) -> None:
        print("Model history reset...")
        self.history = []

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    def _get_object_id(
        self, raw_object_id: str, env_observation: Event
    ) -> Optional[str]:
        visible_objects_ids = [
            o["objectId"] for o in env_observation.metadata["objects"] if o["visible"]
        ]
        for visible_object_id in visible_objects_ids:
            if raw_object_id.lower() in visible_object_id.lower():
                return visible_object_id
        return None

    def postprocess_response(
        self, env_observation: Event, response: str
    ) -> AgentResponse:
        if response.startswith("[Action]"):
            raw_action = response.replace("[Action]", "").strip()
            is_interaction_action = any(
                raw_action.startswith(interaction)
                for interaction in [
                    "OpenObject",
                    "CloseObject",
                    "PickupObject",
                    "PutObject",
                    "DropObject",
                    "ToggleObject",
                    "SliceObject",
                ]
            )
            if is_interaction_action:
                pattern = r"(\w+)\((\w+).*\)"
                match = re.match(pattern, raw_action)
                if match:
                    action = match.group(1)
                    raw_object_id = match.group(2).strip()
                    object_id = self._get_object_id(raw_object_id, env_observation)
                    if object_id is None:
                        return AgentActionResponse(
                            action="No visible object with that id."
                        )
                    return AgentActionResponse(action=action, objectId=object_id)

            main_command = raw_action.split("(")[0]
            if main_command in ("RotateRight", "RotateLeft"):
                degrees = (
                    float(raw_action.split("(")[1].replace(")", ""))
                    if "(" in raw_action
                    else 30.0
                )
                return AgentActionResponse(action=main_command, degrees=degrees)
            return AgentActionResponse(action=main_command)

        verbal_response = response.replace("[Say]", "").strip()
        return AgentTextualResponse(response=verbal_response)

    def act(self, env_observation: Event, language_input: str) -> AgentResponse:
        visible_objects = [
            o["objectId"] for o in env_observation.metadata["objects"] if o["visible"]
        ]
        objects_str = "\n".join(visible_objects)

        try:
            completion = self.client.chat.completions.create(
                temperature=0,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Visible objects: {objects_str}"},
                    {"role": "user", "content": language_input},
                ],
            )
            raw_response = completion.choices[0].message.content
            return self.postprocess_response(env_observation, raw_response)
        except Exception as e:
            print(f"Error: {e}")
            return AgentTextualResponse(response="Error processing request.")
