import base64
import io
import re
from typing import Optional

from openai import OpenAI
from PIL import Image

from responses import AgentActionResponse, AgentResponse, AgentTextualResponse


class VLMClient:
    def __init__(self, model_name="gemma3:4b", system_prompt=None):
        self.model_name = model_name
        self.history = []
        assert system_prompt is not None, "System prompt cannot be None."
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    def reset(self):
        print("Model history reset...")
        self.history = []

    def encode_image(self, image: Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_object_id(self, raw_object_id: str, env_observation) -> Optional[str]:
        visible_objects_ids = [
            o["objectId"] for o in env_observation.metadata["objects"] if o["visible"]
        ]
        for visible_object_id in visible_objects_ids:
            if raw_object_id.lower() in visible_object_id.lower():
                return visible_object_id
        return None

    def postprocess_response(self, env_observation, response) -> AgentResponse:
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
                    raw_object_id = match.group(2)
                    object_id = self._get_object_id(raw_object_id, env_observation)
                    if object_id is None:
                        return AgentActionResponse(
                            action="No visible object with that id."
                        )
                    return AgentActionResponse(action=action, objectId=object_id)

            main_command = raw_action.split("(")[0]
            if main_command in ("RotateRight", "RotateLeft"):
                try:
                    degrees = float(raw_action.split("(")[1].replace(")", ""))
                except:
                    degrees = 30.0
                return AgentActionResponse(action=main_command, degrees=degrees)
            return AgentActionResponse(action=main_command)

        verbal_response = response.replace("[Say]", "").strip()
        return AgentTextualResponse(response=verbal_response)

    def act(self, env_observation, language_input) -> AgentResponse:
        pil_image = Image.fromarray(env_observation.frame)
        base64_image = self.encode_image(pil_image)

        try:
            completion = self.client.chat.completions.create(
                temperature=0,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                            {"type": "text", "text": language_input},
                        ],
                    },
                ],
            )
            raw_response = completion.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error: {e}")

        return self.postprocess_response(env_observation, raw_response)
