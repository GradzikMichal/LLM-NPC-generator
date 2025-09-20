# LLM-NPC-generator
A repository has been created for the purpose of carrying out the recruitment task, which involves using a LLM model for generating custom NPCs.

## Prerequisites

- [Ollama](https://ollama.com/download) should be installed and running
- [Python 3.12](https://www.python.org/downloads) should be installed

## Installation
### Virtual environment
After installing Ollama from link above open folder `LLM-NPC-generator` in terminal and create virtual environment using
command below:
```shell
python -m venv .venv
```
or 
```shell
python3 -m venv .venv
```
depends on your Python setup.

### Installing requirements

In next step one needs to activate virtual environment. Activation depends on user system.

For Linux:
```shell
source .venv/bin/activate
```
For Windows:
```shell
./.venv/Scripts/activate
```

After previous step use:
```shell
pip install -r requirements.txt
```

## Usage
To run program in CLI mode written using functional programming:
```shell
python src/functional/main.py -s [path to story] -m [model name] --model_args [additional model arguments] -r [path to RAG folder] - v [verbose] 
```
To run program in CLI mode written using OOP:
```shell
python src/OOP/main.py -s [path to story] -m [model name] --model_args [additional model arguments] -r [path to RAG folder] - v [verbose] 
```

- Only path to the story is required argument, 
- Model name should be taken from [Ollama page](https://ollama.com/search),
- Additional model arguments should be in form key1=value1 key2=value2 e.g. repeat_last_n=-1,
- Path to the RAG allows one to change RAG type to persistent and choose where RAG database is saved,
- Setting verbose enables more logs.

After model is loaded one can provide prompt to a model or write `quit` to exit application.

## Default settings
- As a default model I chose [llama3.1:8b](https://ollama.com/library/llama3.1) because of the model results and size which allowed me to load whole model on GPU. In my case this model was fast and efficient.
- I have used `num_ctx=40960, repeat_last_n=-1, repeat_penalty=1.2` as a default model parameters. Those allow me to get acceptable results.
- Chroma was chosen as a vector database.
- As an embedding model I used `sentence-transformers/all-mpnet-base-v2`.

## Example
As an example I wrote 4 prompts, one after the other.
### First one:

`"Create a NPC which lives in the world of Sundered Realm. Generate a NPC name, to which faction NPC belong and its personality traits according to the story."`

Model response:

```text
{
  "name": "Kaelin Vexar",
  "faction": "Ethereal Schism",
  "traits": [
    "Enigmatic",
    "Curious",
    "Respectful",
    "Brilliant",
    "Independent"
  ]
}
```

### Second prompt:
`"Tell me which factions are in the world of Sundered Realm."`

Model response:
```text
The factions in the world of Sundered Realm are:

* The Ironhold Clans
* The Verdant Covenant
* The Ashward Collective
* The Ethereal Schism
* The Remnant Assembly

Note: These factions have developed their own distinct cultures, traditions, and ways of seeing the world, often with conflicting values and perspectives.
```

### Third prompt:
`"Create me 10 NPC which lives in the world of Sundered Realm. Generate a NPC name, to which faction NPC belong, what NPC is doing for living and its personality traits according to the story."`

Model response:
```
[
  {
    "name": "Kaelin Vexar",
    "faction": "Ironhold Clans",
    "occupation": "Blacksmith",
    "traits": ["Resilient", "Resourceful", "Loyal"]
  },
  {
    "name": "Aethera Moonwhisper",
    "faction": "Verdant Covenant",
    "occupation": "Wildlife Guide",
    "traits": ["Empathetic", "Adaptable", "Intuitive"]
  },
  {
    "name": "Gorvoth Ironfist",
    "faction": "Ironhold Clans",
    "occupation": "Warrior",
    "traits": ["Brutal", "Ambitious", "Disciplined"]
  },
  {
    "name": "Lysander Starweaver",
    "faction": "Ethereal Schism",
    "occupation": "Scholar",
    "traits": ["Inquisitive", "Analytical", "Enigmatic"]
  },
  {
    "name": "Niamh Ashwood",
    "faction": "Ashward Collective",
    "occupation": "Survivalist",
    "traits": ["Pragmatic", "Resilient", "Independent"]
  },
  {
    "name": "Eira Shadowglow",
    "faction": "Verdant Covenant",
    "occupation": "Healer",
    "traits": ["Compassionate", "Intuitive", "Nurturing"]
  },
  {
    "name": "Thrain Stonebeard",
    "faction": "Ironhold Clans",
    "occupation": "Engineer",
    "traits": ["Practical", "Methodical", "Unyielding"]
  },
  {
    "name": "Cormac Windsong",
    "faction": "Ethereal Schism",
    "occupation": "Musician",
    "traits": ["Creative", "Free-spirited", "Perceptive"]
  },
  {
    "name": "Zhilin Nightwalker",
    "faction": "Ashward Collective",
    "occupation": "Scout",
    "traits": ["Stealthy", "Resourceful", "Self-reliant"]
  },
  {
    "name": "Fianna Dawnrunner",
    "faction": "Remnant Assembly",
    "occupation": "Herald",
    "traits": ["Charismatic", "Diplomatic", "Traditional"]
  }
]
```

#### Forth prompt
`"Tell me something about the cultures in the world of Sundered Realm."`

Model response:
```text
The world of Sundered Realm is characterized by five distinct factions that emerged from the ruins of the great kingdom of Aethermoor. Each faction has developed its own culture, traditions, and ways of survival. The factions are:

* The Ironhold Clans, who retreated to the mountains and value what can be held, measured, and forged.
* The Verdant Covenant, who disappeared into the Deepwood and speak of harmony and adaptation.
* The Ashward Collective, who remained in the blast lands closest to the original catastrophe and are survivors above all else.
* The Ethereal Schism, who established sanctuaries in the spaces between territories and seek understanding of what was broken and what might be mended.
* The Remnant Assembly, who hold the coastal strongholds and maintain the old ceremonies and hierarchies.

These factions have developed different stories about the Sundering, which occurred three centuries ago when a ritual gone wrong shattered the very fabric that held the realm together. The Sundering has had a profound impact on the world, causing magic to pool in dangerous concentrations in some places while leaving others barren. The world is now at a crossroads, with small choices having the potential to ripple outward in unexpected ways.

```

### Results analysis
In my opinion results are good, model is not hallucinating. In the response for the third prompt, one can see a problem. 
The name of the first character is repeated from the first prompt but character itself is not repeated wholly. 
This can be seen as two different persons as in real live.

Other than this model provides results from `fantasy.md` so the RAG is working and helping the model.

---
Additionally, I have created basic GUI for my own fun and learning process. It works almost exactly the same as CLI version, in this case user must provide story path and chose or write model name.
