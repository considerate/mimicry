import pyglet
pyglet.options["headless"] = True
from mimicry.mimicry import main as cars

def main():
    cars(headless=True)
