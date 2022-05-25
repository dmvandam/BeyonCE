from GridPropertyName import GridPropertyName


def test_disk_radius() -> None:
    parameter = GridPropertyName.DISK_RADIUS
    name = parameter.__str__()
    assert name == "Disk Radius"


def test_inclination() -> None:
    parameter = GridPropertyName.INCLINATION
    name = parameter.__str__()
    assert name == "Inclination"


def test_tilt() -> None:
    parameter = GridPropertyName.TILT
    name = parameter.__str__()
    assert name == "Tilt"


def test_fx_map() -> None:
    parameter = GridPropertyName.FX_MAP
    name = parameter.__str__()
    assert name == "Fx Map"


def test_fy_map() -> None:
    parameter = GridPropertyName.FY_MAP
    name = parameter.__str__()
    assert name == "Fy Map"


def test_diagnostic_map() -> None:
    parameter = GridPropertyName.DIAGNOSTIC_MAP
    name = parameter.__str__()
    assert name == "Diagnostic Map"


def test_gradient() -> None:
    parameter = GridPropertyName.GRADIENT
    name = parameter.__str__()
    assert name == "Gradient"


def test_gradient_fit() -> None:
    parameter = GridPropertyName.GRADIENT_FIT
    name = parameter.__str__()
    assert name == "Gradient Fit"