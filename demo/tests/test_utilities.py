from tlc.utilities import all_equal


def test_all_equal():
    a = [1, 2, 3]
    b = [2, 2, 2, 2]

    assert not all_equal(a)
    assert all_equal(b)
