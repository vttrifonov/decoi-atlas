#%%
if __name__ == '__main__':
    __package__ = 'decoi_atlas'

from pathlib import Path

#%%

class _config:
    project='decoi-atlas'
    cache = Path.home()/'.cache'/project
    root = Path(__file__).parent

config = _config()

