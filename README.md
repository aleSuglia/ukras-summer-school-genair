# UK Robotics Summer School 2025 -- Generative AI for Robotics

The UK Robotics Summer School (UKRSS) 2025 focuses on cutting-edge topics in robotics, including Generative AI for Robotics. Hosted by Heriot-Watt University, this summer school provides participants with hands-on tutorials and lectures to explore the intersection of robotics and artificial intelligence. For more details, visit [UKRSS 2025](https://ukrss.site.hw.ac.uk/).

## Requirements

Before running the tutorial, ensure the following dependencies are installed:

### 1. Ollama
Ollama is required for managing AI models. Follow the steps below to install it:
- Visit the [Ollama website](https://ollama.ai/) and download the appropriate installer for your operating system.
- Follow the installation instructions provided on the website.
- Once you have installed Ollama, you need to download the models we recommend for this tutorial
  using the following commands:
    - **Qwen2-1.5B**: `ollama pull qwen2.5:1.5b` (you can read more about this model [here](https://arxiv.org/abs/2412.15115))
    - **Gemma3-4B**: `ollama pull gemma3:4b` (you can read more about this model [here](https://blog.google/technology/developers/gemma-3/))

### 2. Python
Python is used to run the tutorial scripts. To avoid potential issues, it is recommended to install Python using [Anaconda](https://www.anaconda.com/):
- Download and install Anaconda for your operating system.
- Create a new environment with Python 3.11:
    ```bash
    conda create -n embodied-ai python=3.11
    conda activate embodied-ai
    ```
- Verify the installation by running:
    ```bash
    python --version
    ```

### 3. Additional Python Libraries
Install the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

## Embodied AI Tutorial

This tutorial introduces participants to Embodied AI concepts and their applications in robotics.

This codebase comes bundled with two different scripts that allow you to test different Large
Language Models (*text-only*) or Vision-and-Language Models (VLMs)  and
how they can operate in a 3D simulated environment like [AI2Thor](https://ai2thor.allenai.org/).
We provide a description of the scripts below:

- [genair_llm.py](genair_llm.py): runs a chat session with an LLM that can interact with the
  AI2Thor home environment
- [genair_vlm.py](genair_vlm.py): runs a chat session with a VLM that can interact with the AI2Thor
  home environment

Both scripts require that you have started Ollama in a separate terminal window. You can do so by calling the command:

  ```bash
  ollama serve
  ``` 
Once you do that, Ollama will be able to load models which will respond to requests you send via
each Python program above (*do not close this terminal window because otherwise your LLMs won't be
able to communicate with the Python scripts we have created*)

### [Task 0] Familiarise yourself with the code

For the majority of this tutorial, we will use the script [genair_llm.py](genair_llm.py). Start
exploring the code we have created for you. We have left some useful comments and marked some parts
of it with `TODO` comments to indicate parts that require you to add/complete them with some code.

### [Task 1] Let's start exploring LLM for Embodied AI

Start with the following system prompt, and test whether the system can a) carry out the actions
described in the prompt, b) respond to chit-chat (e.g. "How are you?") , c) answer general
knowledge questions (e.g. "What is france"?).  

You can play a video of your interaction using the command: *ffplay rollout.mp4*. 

Note down any errors that you encounter. 

```python
SYSTEM_PROMPT = """ 

You are an embodied agent that receives images and acts in a 3D simulated environment.  You can move around and interact with objects. These are the actions at your disposal: 

- MoveAhead: agent moves ahead one step 

- MoveBack: agent moves back one step 

You can also use manipulation actions which require you to specify the object name of a visible object.  

OpenObject(<object name>): agent opens the object 

If you generate an action, start your response with the tag `[Action]` and follow the format of the
action. Never put quotes around the object name. For example: [Action] OpenObject(Fridge)",   
"""
``` 

### [Task 2] Now let's add some navigation actions to our agent

Given the simulator [API](https://ai2thor.allenai.org/) (which describes actions such as moving, rotating, and looking) add to the prompt above so that the robot can perform the following actions: 

- move left
- move right
- rotate right
- rotate left
- look up
- look down

### [Task 3] Adding object manipulation skills to our agent

Based on your prompt from step 2 and the simulator [API](https://ai2thor.allenai.org/), add to the prompt so that the robot can manipulate named objects in the following ways:  

- close object
- pick up object
- drop object
- put object in another object
- switch object on/off
- slice object (which requires a knife) 

 Test that your robot is able to carry out all of the above actions.

### [Task 4] Create an evaluation set

Create a test set of inputs (at least 10) that you expect your robot to be able to handle. For each
item in the test set this should be: user input + expected robot output (action and/or speech).

Use this evaluation set as a benchmark for your agent. What is the success rate of your agent?

### [Task 5] Test another LLM

Thanks to Ollama, we have access to a variety of state-of-the-art LLMs that we can directly use in
our laptop. At the moment, we've been using **Qwen2.5-1.5B**. Go to the [Ollama
models](https://ollama.com/search) and find a suitable candidate model. To use it, ensure you have
downloaded it as described in the [Requirements](#requirements) section. You can find an example of
how to integrate this model in the codebase at [line 330 of genair_llm.py](genair_llm.py#L330).

Test this new LLM: is it better or worse than the previous one?

### [Task 6] Using a VLM

In this task, you will use a local Vision-Language Model (VLM) to process and analyze 
a picture of the simulator's screen. The purpose of this task is to determine 
whether the visual and textual information extracted from the screen aids in 
understanding the instructions provided in the evaluation set. We will use the script
[genair_vlm.py](genair_vlm.py) for this exercise. This is a similar script to the previous one with
the difference that it has already functionalities to process image and text.

Test this model on your evaluation set. What is your success rate?

### [Task 7] Compile a table of results 

Now that you have explored a variety of different models, just like in a research paper, you should
compile a table of results that compares your models. Write down a table and discuss your findings.

## [Optional] Advanced exercises

Well done for completing all the exercises so far. This set of exercises are optional and allow you
to explore more sophisticated concepts in this space. 

### [Advanced Task 1] 
Edit the code and prompt so that the robot always responds with a report of what it is doing (e.g.
"OK. I am putting the cup in the microwave") and when it has completed an action (e.g. "Ok, the cup
is now inside the microwave"). Is it too verbose or too concise? Try to change the output length
and style (e.g. "use slang in your responses").

### [Advanced Task 2]

Test whether your robot can: 

- execute complex commands (e.g. "Do X then Y then Z")  
 
- execute high-level commands (e.g. "Make a cup of tea", "Tidy up")
  
If it cannot,  add elements to the code and prompt that enable this. 
