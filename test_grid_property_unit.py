from GridPropertyUnit import GridPropertyUnit


def test_eclipse_duration_unit() -> None:
    parameter = GridPropertyUnit.ECLIPSE_DURATION
    assert parameter.get_unit() == "$t_{ecl}$"


def test_eclipse_duration() -> None:
    parameter = GridPropertyUnit.ECLIPSE_DURATION
    name = parameter.__str__()
    assert name == "Eclipse Duration"


def test_degree_unit() -> None:
    parameter = GridPropertyUnit.DEGREE
    assert parameter.get_unit() == "$^o$"


def test_degree() -> None:
    parameter = GridPropertyUnit.DEGREE
    name = parameter.__str__()
    assert name == "Degree"


def test_none_unit() -> None:
    parameter = GridPropertyUnit.NONE
    assert parameter.get_unit() == "-"


def test_eclipse_none() -> None:
    parameter = GridPropertyUnit.NONE
    name = parameter.__str__()
    assert name == "None"