import pytest

from rtv_solver.structure.node import Node

@pytest.mark.basic
@pytest.mark.parametrize(
    "node1, node2, expected",
    [
        pytest.param(Node(34.0, -118.0), Node(34.0, -118.0), True),
        pytest.param(Node(34.0, 120.0), Node(34.0, -118.0), False),
        pytest.param(Node(34.0, -118.1), Node(34.0, -118.0), False),
    ])
def test_node_equality(node1, node2, expected):
    assert (node1 == node2) is expected

def test_node_from_dict():
    data = {"lat": 48.1351, "lon": 11.5820}
    node = Node.from_dict(data)
    assert node.lat == 48.1351
    assert node.lon == 11.5820
    assert node.id is None

def test_node_copy_independence():
    original = Node(34.0, -118.0, id=1)
    cloned = original.copy()
    
    assert original == cloned
    assert original is not cloned
    assert cloned.id == 1