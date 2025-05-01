import base64
import io
import re
from typing import Optional

import imageio
import numpy
from ai2thor.controller import Controller
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

SYSTEM_PROMPT = """
You are an embodied agent that receives images and acts in a 3D simulated environment. Your task is
to follow the instructions provided by the user. 
You can move around and interact with objects. These are the actions at your disposal:
- MoveAhead: agent moves ahead one step
- MoveBack: agent moves back one step
- MoveLeft: agent moves to the left one step
- MoveRight: agent moves to the right one step
- RotateRight(degrees): agent rotates to the right by a given number of degrees
- RotateLeft(degrees): agent rotates to the left by a given number of degrees
- LookUp: agent looks up
- LookDown: agent looks down
You can also use manipulation actions which require you to specify the object name of a visible object:
- OpenObject(<object name>): agent opens the object
- CloseObject(<object name>): agent closes the object
- PickupObject(<object name>): agent picks up the object and places it in the inventory
- PutObject(<receptacle name>): the agent has an object in the inventory and places it in the receptacle 
- DropObject(<object name>): the agent drops the object
- ToggleObjectOn(<object name>): the agent toggles the object on
- ToggleObjectOff(<object name>): the agent toggles the object off
- SliceObject(<object name>): the agent slices the object (requires a knife)
If you generate an action, start your response with the tag `[Action]` followed by
`<Action>(<object name>)`. 
Respond to the instruction with the action you would like to take.
",
"""

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


class AgentResponse(BaseModel):
    def to_dict(self):
        current_dict = self.model_dump(mode="python")

        return {key: value for key, value in current_dict.items() if value is not None}


class AgentTextualResponse(AgentResponse):
    response: str


class AgentActionResponse(AgentResponse):
    action: str
    objectId: Optional[str] = None
    degrees: Optional[float] = None


class VLMClient:
    # You can pass in the model name as a string
    # make sure that you "pull" the model first using ollama pull <model_name>
    def __init__(self, model_name="moondream"):
        self.model_name = model_name
        self.history = []

    def reset(self):
        print("Model history reset...")
        self.history = []

    def encode_image(self, image: Image):
        # Convert the PIL image to a byte stream
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        # Encode the byte stream to base64
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_object_id(self, raw_object_id, env_observation):
        visible_objects_ids = [
            o["objectId"] for o in env_observation.metadata["objects"] if o["visible"]
        ]
        object_id = None

        for visible_object_id in visible_objects_ids:
            if raw_object_id in visible_object_id:
                object_id = visible_object_id
                break

        return object_id

    def postprocess_response(self, env_observation, response):
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
                # create regex to extract object id and action name based on template
                # e.g. OpenObject(<object_id>)

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

            # now extract the first part only
            main_command = raw_action.split("(")[0]
            if main_command in ("RotateRight", "RotateLeft"):
                # attempt to parse the degrees from the raw action
                degrees = None
                try:
                    degrees = float(raw_action.split("(")[1].replace(")", ""))
                except:
                    degrees = 30.0
                # if degrees is not provided, default to 30.0
                return AgentActionResponse(action=main_command, degrees=degrees)
            return AgentActionResponse(action=main_command)

        else:
            verbal_response = response.replace("[Say]", "").strip()
            return AgentTextualResponse(response=verbal_response)

    def act(self, env_observation, language_input):
        pil_image = Image.fromarray(env_observation.frame)
        base64_image = self.encode_image(pil_image)

        try:
            completion = client.chat.completions.create(
                temperature=0,
                model=self.model_name,
                messages=[
                    # {"role": "system", "content": SYSTEM_PROMPT},
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
                # response_format=AgentResponse,
            )

            raw_response = completion.choices[0].message.content
            # if action_response.parsed:
            #     print(action_response.parsed)
            # elif action_response.refusal:
            #     print(action_response.refusal)
        except Exception as e:
            raise ValueError(f"Error: {e}")

        return self.postprocess_response(env_observation, raw_response)


def main():
    controller = Controller(
        quality="Medium", renderDepthImage=False, width=640, height=640
    )

    client = VLMClient()

    frames = []
    event = controller.step("Initialize")
    image = Image.fromarray(event.frame)
    frames.append(image)

    done = False

    while not done:
        language_instruction = input("Enter instruction (enter CLOSE to exit): ")
        if language_instruction == "CLOSE":
            print("Interrupting task...")
            break

        print("Processing instruction...")
        response = client.act(event, language_instruction)
        print(f"Generated action: {response}")
        try:
            event = controller.step(**response.to_dict(), forceAction=True)
            # forces the GUI to update
            controller.step("NoOp")
            image = Image.fromarray(event.frame)
            frames.append(image)
        except Exception as e:
            print(f"Unable to execute action due to an error: {e}")

    # Encode all frames into a mp4 video.
    video_path = "rollout.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=1, codec="libx264")

    print(f"Video of the evaluation is available in '{video_path}'.")


if __name__ == "__main__":
    main()
