<h3>MARIO-BOT-NEAT</h3>
<h6>A simple Mario bot using NEAT GA</h6>

Main requirements
- python3
- pip3
- SuperMarioBros ROM (provided)
- fceux (apt-install)

Create a virtual environment if necessary
- python3 -m venv .env; source .env/bin/activate
Execute : pip3 install -r requirements.txt
Note : For older system, manual install requirements of older versions

Note : gym-retro might require to install via source files : <a href="https://github.com/openai/retro">Here</a>

To install the SuperMarioBros ROM (requires manual installation):
- File : rom.nes
- Dest : <" directory to gym-retro ">/gym-retro/retro/data/stable/SuperMarioBros-Nes/
Note : the directory to retro varies according to system (For Debain/Ubuntu env : /usr/local/lib/python3.6/dist-packages/) or if installed via source files, go to location of the gitclone.

Run : python3 mario.py
Note : provided with the file are the pre-excuted checkpoints (only till gen99) for easier completion of training. If you wish to train from the first set line 10 to False.