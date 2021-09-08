# NTT-Internship
Summer 2021. Simulating excitons in monolayer TMDs under electric and magnetic fields.

The user can choose various thingsm for example a uniform or exponentially spaced grid, and a Coulomb or Keldysh potential. The Keldysh is well known to be more accurate for very thin (quasi-2D) materials. All of these choices, along with the material parameters, are defined in INPUT.txt.

The user can also choose to add a uniform in-plane electric field, out-of-plane magnetic field or both simultaneously (currently only constant E field with varying B field, which is usually the experimentally relevant option).

The main code currently (as of Wed 8/9/21) is Excitons_in_TMDs.py, which uses INPUT.txt, Finite_differences.py, Potentials.py, Grids_2D.py, Symmetrise_and_solve.py and Plotting.py.

INPUT_explanantions.txt explains all the input parameters in INPUT.txt.

There were various old scripts (beginning from simple models of the hydrogen atom) which I have now deleted from here.


