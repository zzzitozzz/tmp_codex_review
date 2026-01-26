from model import Rect


def test_rect_contains_includes_edges():
    rect = Rect(1.0, 2.0, 3.0, 4.0)

    assert rect.contains((1.0, 3.0))
    assert rect.contains((2.0, 4.0))
    assert rect.contains((1.5, 3.5))


def test_rect_contains_excludes_outside():
    rect = Rect(1.0, 2.0, 3.0, 4.0)

    assert not rect.contains((0.9, 3.5))
    assert not rect.contains((2.1, 3.5))
    assert not rect.contains((1.5, 2.9))
    assert not rect.contains((1.5, 4.1))
