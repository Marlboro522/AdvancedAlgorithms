from src.rough import *


def test1():
    source, target = 7153623736, 555759091
    assert custom_variant(G, source, target) == shortest_path(G, source, target)
    
