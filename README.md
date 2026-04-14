# FraNC - Framework for Noise Cancellation in gravitational wave detection

![Test status](https://github.com/NewtonianNoise/franc/actions/workflows/testing.yml/badge.svg)
![Linting status](https://github.com/NewtonianNoise/franc/actions/workflows/pylint.yml/badge.svg)
![Static type check status](https://github.com/NewtonianNoise/franc/actions/workflows/mypy.yml/badge.svg)

A framework to develop and evaluate noise cancellation techniques.
Includes python implementations of different static and adaptive filtering techniques.
The techniques for the prediction of a correlated signal component from witness signals provide a unified interface.

[Documentation](https://franc.readthedocs.io/en/latest/), [Development guide](DEVELOPMENT.md), [Contributors](CONTRIBUTORS.md)

## Install

From pypi: `pip install franc`

From repository: `pip install .`

From repository (editable): `pip install hatchling ninja && make ie`

## Compatibility

This package is intended to be used with a recent `numpy` release.
Support for `numpy` back to `1.26.4` is tested and should work.

Automated checks during the development process are only performed for `linux`.
That makes it more likely for issues to slip through on `windows`, so switching to `linux` or `mac` might be a solution to resolve issues.
Please open an entry on the github issue tracker if you find something that does not work.

## License

Copyright (C) 2025  Tim J. Kuhlbusch et al.

```
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
```
