from src.main import shortest_path


def test_shortest_path():
    graph = [[1, 2], [0, 2, 3], [0, 1],[1]]
    start = 0
    end = 3
    assert shortest_path(graph, start, end) == 2

