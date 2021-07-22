# NTT-Internship
Summer 2021, "enhancing polariton-polariton interactions via a mediating material"

Starting with a simulation of excitons in TMDs - using a finite differences scheme. 
The user can choose a uniform or exponentially spaced grid, and a Coulomb or Keldysh potential. The Keldysh is well known to be more accurate for very thin (quasi-2D) materials.
The user can also choose to add a uniform in-plane electric field or perpendicular magnetic field.

The main code currently (as of 22/7/21) is Excitons_in_TMDs.py, which uses Finite_differences.py, Potentials.py and Grids_2D.py.
The other scripts (begining GaAs... or Hydrogen...) are earlier versions and are now somewhat redundant (unless you actually want to simulate a Hydrogen atom of course).


