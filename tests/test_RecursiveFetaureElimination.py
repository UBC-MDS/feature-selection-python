from feature_selection import recursive_feature_elimination


def test_1():
    result = recursive_feature_elimination(1, 2, 3, 4, 5)
    assert result == []
