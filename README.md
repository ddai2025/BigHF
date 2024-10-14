# BigHF
Two-dimensional Hartree-Fock code for a large torus without lattice symmetry.

Because this code does not impose any lattice symmetry, it is ideal for "discovery" calculations in which one does not know what the candidate ground states are. This code uses a direct minimization algorithm instead of the more common self-consistent field method. In direct minimization, the Hartree-Fock energy is analytically continued onto the space of nonorthonormal orbitals using Lowdin symmetric orgonalization, bypassing the need for Lagrange multipliers. The resulting _unconstrained_ minimization problem can be attacked with standard techniques such as nonlinear conjugate gradient.

I developed this method for Ref [1] and provide detailed derivations its Supplementary Material. The direct minimization algorithm was originally developed for density functional theory in Ref [2].

References
1. Strong-Coupling Phases of Trions and Excitons in Electron-Hole Bilayers at Commensurate Densities. David D. Dai and Liang Fu. Phys. Rev. Lett. 132, 196202 (2024).
2. Ab initio molecular dynamics: Analytically continued energy functionals and insights into iterative solutions. T. A. Arias, M. C. Payne, and J. D. Joannopoulos. Phys. Rev. Lett. 69, 1077 (1992).
