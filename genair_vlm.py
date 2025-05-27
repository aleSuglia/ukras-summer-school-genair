import imageio
import numpy
from ai2thor.controller import Controller
from PIL import Image

from responses import AgentTextualResponse
from utils import render_text_on_image
from vlm_client import VLMClient

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
If you generate an action, start your response with the tag `[Action]` and follow the format of the
action. 
When answering a question, use the tag `[Say]` to generate a verbal response.",
"""


def main():
    controller = Controller(
        quality="Medium", renderDepthImage=False, width=640, height=640
    )

    model_name = "gemma3:4b"
    client = VLMClient(model_name=model_name, system_prompt=SYSTEM_PROMPT)

    frames = []
    event = controller.step("Initialize")
    image = Image.fromarray(event.frame)
    frames.append(image)

    done = False

    while not done:
        image = Image.fromarray(event.frame)
        frames.append(image)
        language_instruction = input("Enter instruction (enter CLOSE or x to exit): ")
        if language_instruction in ("CLOSE", "x"):
            print("Interrupting task...")
            break

        text_image = render_text_on_image(language_instruction, 640, 640)
        frames.append(text_image)

        print("Processing instruction...")
        response = client.act(event, language_instruction)
        if isinstance(response, AgentTextualResponse):
            print(f"Generated response: {response.response}")
            robot_text_image = render_text_on_image(
                response.response.replace(".", "\n"), 640, 640
            )
            frames.append(robot_text_image)
        else:
            print(f"Generated action: {response}")
            try:
                event = controller.step(**response.to_dict(), forceAction=True)
                controller.step("NoOp")
            except Exception as e:
                print(f"Unable to execute action due to an error: {e}")

    video_path = "rollout.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=1, codec="libx264")
    print(f"Video of the evaluation is available in '{video_path}'.")


if __name__ == "__main__":
    main()
