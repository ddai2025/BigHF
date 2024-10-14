"""
Numbers complex unless noted otherwise.
"""


import sys
import jax
from jax import jit, vmap
from jax.numpy import *
from functools import partial
from jax.scipy.linalg import sqrtm
from jax.numpy.linalg import pinv, eigh
import numpy.random as random
from time import perf_counter
from scipy import optimize
from scipy.linalg import eigh
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
prec=complex64


"""
Helper functions for analytically continued energy functional.
"""
@jit
def lowdin(C: ndarray) -> ndarray:
    """
    Performs lowdin symmetric orthogonalization.

    Parameters
    ----------
    C : ndarray
        Arbitrary (non-ON) coefficient matrix.

    Returns
    -------
    C_ortho : ndarray
        Closest ON coefficient matrix.
    """
    S = C.T.conj() @ C
    C_ortho = C @ sqrtm(pinv(S))
    return (C_ortho)

@jit
def S_inv_and_D_flavors(C_flavors: list[ndarray]) \
    -> tuple[list[ndarray], ndarray]:
    """
    Get inverse of overlap matrix and post-orthogonalization density matrix
    for each flavor.

    Parameters
    ----------
    C_flavors : list
        Coefficient matrix for each flavor.

    Returns
    -------
    S_inv_flavors : list
        List of inverse overlap matrix for each flavor.
    D_flavor : ndarray
        Density matrix for each flavor, stored as a rank-3 tensor with 
        shape (N_flavors, N_basis, N_basis).
    """
    N_flavors = len(C_flavors)
    N_basis = C_flavors[0].shape[0]
    D_flavors = zeros((N_flavors, N_basis, N_basis), dtype=C_flavors[0].dtype)
    S_inv_flavors = []

    for i, C in enumerate(C_flavors):
        S = C.T.conj() @ C
        S_inv = pinv(S)
        D = (C @ S_inv @ C.T.conj()).T
        D_flavors = D_flavors.at[i].set(D)
        S_inv_flavors.append(S_inv)
    return ((S_inv_flavors, D_flavors))

@jit
def gradient(C: ndarray, S_inv: ndarray, F: ndarray) -> ndarray:
    """
    Gradient of energy (analytically continued onto the space of non-ON
    orbitals using Lowdin symmetric orthogonalization) with respect to the
    coefficient matrix of a specific flavor.

    Parameters
    ----------
    C : ndarray
        Arbitrary (non-ON) coefficient matrix.
    S_inv : ndarray
        Inverse of overlap matrix.
    F_flavors : ndarray
        Fock matrix for each flavor

    Returns
    -------
    dE_dC : ndarray
        Gradient of the energy
    """
    N_basis, _ = C.shape
    first_part = eye(N_basis, dtype=C.dtype) - C @ S_inv @ C.T.conj()
    second_part = F @ C @ S_inv
    dE_dC = 2 * first_part @ second_part # See SM for factor of 2.
    return (dE_dC)

@jit
def flatten(C_flavors: list[ndarray]) -> ndarray:
    """
    Flatten the list of coefficient matrices into a real vector.
    
    Parameters
    ----------
    C_flavors : list
        Coefficient matrix for each flavor.

    Returns
    -------
    C : ndarray
        Real vector describing HF ansatz.
    """
    
    C_complex = concatenate(C_flavors, axis=None)
    C = concatenate((C_complex.real, C_complex.imag))
    return (C)


class BigTorus():
    """
    Plane-wave Hartree-Fock for a large torus without lattice symmetry.
    """
    def __init__(self, L_x: float, L_y: float, n_x: int, n_y: int,
                 flavors: list[str], pop_flavors: ndarray,
                 m_flavors: ndarray, v_functions: list[list[callable]]):
        """
        Initialize HF system.

        Parameters
        ----------
        L_x : int
            Torus x-dimension.
        L_y : int
            Torus y-dimension.
        n_x : int
            Plane-wave grid has k_x = 2 * pi * [-n_x, ..., n_x] / L_x
        n_y : int
            Plane-wave grid has k_y = 2 * pi * [-n_y, ..., n_y] / L_y
        flavors : list
            Names of flavors, for example ["electron", "hole"].
        pop_flavors : ndarray
            Number of particles for each flavor.
        m_flavors : ndarray
            m_flavors[i, u] = effective mass of species (i) in direction (u).
        v_functions : list[list] 
            v_functions[i][j] = Fourier transform of interaction between
            species (i) and (j).
        """

        # Important constants.
        N_flavors = len(flavors)
        rho_flavors = pop_flavors / L_x / L_y
        k_F_flavors = sqrt(4 * pi * rho_flavors)  # Fermi wavevector
        k_F_max = amax(k_F_flavors)  # Fermi sea convergence
        r_s_flavors = 1 / sqrt(pi * rho_flavors)  # Wigner-Seitz radii
        k_W_max = 1 / amin(r_s_flavors)  # Wigner crystal convergence

        # Initialize basis.
        N_basis = (2 * n_x + 1) * (2 * n_y + 1)  # Basis size
        n_vecs = mgrid[-n_x:n_x + 1, -n_y:n_y + 1].reshape((2, N_basis)).T
        k_vecs = n_vecs * (2 * pi / array([L_x, L_y]))
        k_vecs_norms = sqrt(sum(k_vecs ** 2, axis=1))
        max_k_basis = max(k_vecs_norms)  # Maximum wavevector size included.
        k1p_x, k1p_y, k1_x, k1_y = mgrid[0:2 * n_x + 1, 0:2 * n_y + 1,
                                         0:2 * n_x + 1, 0:2 * n_y + 1]

        # Initialize kinetic energy.
        T_diag = einsum("ui,fi->fu", (1 / 2) * (k_vecs ** 2),
                        1 / m_flavors, optimize="greedy")
        identity = eye((2 * n_x + 1) * (2 * n_y + 1), dtype=complex128)
        T_flavors = einsum("fu,uv->fuv", T_diag, identity, optimize="greedy")
        lowest_pop_flavors = [] # For building the free Fermi sea.
        for flavor in range(N_flavors):
            ordering = argsort(T_diag[flavor, :])
            lowest_pop_flavors.append(ordering[:pop_flavors[flavor]])

        # Save basis data.
        self.L_x = L_x
        self.L_y = L_y
        self.n_x = n_x
        self.n_y = n_y
        self.N_basis = N_basis
        self.k_vecs = k_vecs
        self.max_k_basis = max_k_basis
        self.k1p_x = k1p_x
        self.k1p_y = k1p_y
        self.k1_x = k1_x
        self.k1_y = k1_y

        # Save particle data.
        self.flavors = flavors
        self.N_flavors = N_flavors
        self.pop_flavors = pop_flavors
        self.m_flavors = m_flavors
        self.v_functions = v_functions
        self.T_flavors = T_flavors
        self.lowest_pop_flavors = lowest_pop_flavors

        # Initialize caches.
        self.D_cache = zeros((N_flavors, N_basis, N_basis), complex128)
        self.F_cache = zeros((N_flavors, N_basis, N_basis), complex128)
        self.fock_calls = 0


    """
    Functions to calculate the Fock matrix.
    """
    def coulomb_setup(self):
        """
        Precompute slicers and matrix elements for Coulomb term.
        """
        print("Setting up Coulomb function ...", end=" "); sys.stdout.flush()

        n_x = self.n_x
        n_y = self.n_y
        L_x = self.L_x
        L_y = self.L_y
        N_flavors = self.N_flavors
        v_functions = self.v_functions

        # Slicers for Coulomb D_shifted computation.
        q_x, q_y, k2_x, k2_y = mgrid[-2 * n_x:2 * n_x + 1,
                                     -2 * n_y:2 * n_y + 1,
                                     0:2 * n_x + 1,
                                     0:2 * n_y + 1]
        k2p_x, k2p_y = k2_x - q_x, k2_y - q_y
        cutter_k2p_x = (k2p_x > 2 * n_x) + (k2p_x < 0)
        cutter_k2p_y = (k2p_y > 2 * n_y) + (k2p_y < 0)
        k2p_x_cut = (1 - cutter_k2p_x) * k2p_x + cutter_k2p_x * (2 * n_x + 1)
        k2p_y_cut = (1 - cutter_k2p_y) * k2p_y + cutter_k2p_y * (2 * n_y + 1)

        # Interaction Fourier transforms for Coulomb computation.
        q = sqrt((q_x[:, :, 0, 0] * (2 * pi / L_x)) ** 2 +
                 (q_y[:, :, 0, 0] * (2 * pi / L_y)) ** 2)
        v_q_flavors = zeros((N_flavors, N_flavors, 4 * n_x + 1, 4 * n_y + 1))
        for i in range(N_flavors):
            for j in range(N_flavors):
                v_q_ij = v_functions[i][j](q) / L_x / L_y
                v_q_flavors = v_q_flavors.at[i, j, :, :].set(v_q_ij)
        v_q_flavors = nan_to_num(v_q_flavors, nan=0, posinf=0, neginf=0)

        # Log.
        self.k2_x = k2_x
        self.k2_y = k2_y
        self.k2p_x_cut = k2p_x_cut
        self.k2p_y_cut = k2p_y_cut
        self.v_q_flavors = v_q_flavors
        print("done!")

    @partial(vmap, in_axes=(None, 0, 0, 0), out_axes=0)
    @partial(jit, static_argnums=0)
    def coulomb(self, i: int, j: int, D: ndarray) -> ndarray:
        """
        Compute part of Coulomb matrix.

        Parameters
        ----------
        i : int
            Flavor index.
        j : int
            Flavor index.
        D : int
            Density matrix of species (j).

        Returns
        -------
        J: ndarray
            Contribution of flavor (j) to (i)'s Coulomb matrix.
        """

        n_x = self.n_x
        n_y = self.n_y
        N_basis = self.N_basis
        k2p_x_cut = self.k2p_x_cut
        k2p_y_cut = self.k2p_y_cut
        k2_x = self.k2_x
        k2_y = self.k2_y
        v_q_flavors = self.v_q_flavors
        k1p_x = self.k1p_x
        k1p_y = self.k1p_y
        k1_x = self.k1_x
        k1_y = self.k1_y

        # Accelerate using momentum conservation.
        D = reshape(D, (2 * n_x + 1, 2 * n_y + 1, 2 * n_x + 1, 2 * n_y + 1))
        D_padded = pad(D, ((0, 1), (0, 1), (0, 0), (0, 0)))
        D_shifted = D_padded[k2p_x_cut, k2p_y_cut, k2_x, k2_y]
        J_shifted = einsum("ab,abcd->ab", v_q_flavors[i, j], D_shifted,
                           optimize="greedy")
        J = J_shifted[k1p_x - k1_x + 2 * n_x, k1p_y - k1_y + 2 * n_y]
        J = reshape(J, (N_basis, N_basis))
        return (J)

    def exchange_setup(self):
        """
        Precompute slicers and matrix elements for Coulomb term.
        """
        print("Setting up exchange function ...", end=" "); sys.stdout.flush()

        n_x = self.n_x
        n_y = self.n_y
        L_x = self.L_x
        L_y = self.L_y
        N_flavors = self.N_flavors
        v_functions = self.v_functions

        # Slicers for exchange D_shifted computation
        k_x, k_y, p1_x, p1_y = mgrid[0:2 * n_x + 1, 0:2 * n_y + 1,
                                     0:2 * n_x + 1, 0:2 * n_y + 1]
        p2_x, p2_y, r_x, r_y = mgrid[0:2 * n_x + 1, 0:2 * n_y + 1,
                                     -2 * n_x:2 * n_x + 1,
                                     -2 * n_y:2 * n_y + 1]
        p2r_x, p2r_y = p2_x - r_x, p2_y - r_y
        cutter_p2r_x = (p2r_x > 2 * n_x) + (p2r_x < 0)
        cutter_p2r_y = (p2r_y > 2 * n_y) + (p2r_y < 0)
        p2r_x_cut = (1 - cutter_p2r_x) * p2r_x + cutter_p2r_x * (2 * n_x + 1)
        p2r_y_cut = (1 - cutter_p2r_y) * p2r_y + cutter_p2r_y * (2 * n_y + 1)

        # Interaction Fourier transforms for exchange computation.
        kp1 = sqrt(((2 * pi / L_x) * (k_x - p1_x)) ** 2 +
                   ((2 * pi / L_y) * (k_y - p1_y)) ** 2)
        v_kp1_flavors = zeros((N_flavors, 2 * n_x + 1, 2 * n_y + 1,
                               2 * n_x + 1, 2 * n_y + 1))
        for i in range(N_flavors):
            v_kp1_i = v_functions[i][i](kp1) / L_x / L_y
            v_kp1_flavors = v_kp1_flavors.at[i, :, :, :, :].set(v_kp1_i)
        v_kp1_flavors = nan_to_num(v_kp1_flavors, nan=0, posinf=0, neginf=0)

        # Log.
        self.p2_x = p2_x
        self.p2_y = p2_y
        self.p2r_x_cut = p2r_x_cut
        self.p2r_y_cut = p2r_y_cut
        self.v_kp1_flavors = v_kp1_flavors
        print("done!")

    @partial(vmap, in_axes=(None, 0, 0), out_axes=0)
    @partial(jit, static_argnums=0)
    def exchange(self, i: int, D: ndarray) -> ndarray:
        """
        Compute the exchange matrix.

        Parameters
        ----------
        i : int
            Flavor index.
        D : ndarray
            Density matrix of species (i).

        Returns
        -------
        K : ndarray
            Flavor (i)'s exchange matrix.
        """

        n_x = self.n_x
        n_y = self.n_y
        N_basis = self.N_basis
        p2r_x_cut = self.p2r_x_cut
        p2r_y_cut = self.p2r_y_cut
        p2_x = self.p2_x
        p2_y = self.p2_y
        v_kp1_flavors = self.v_kp1_flavors
        k1p_x = self.k1p_x
        k1p_y = self.k1p_y
        k1_x = self.k1_x
        k1_y = self.k1_y

        # Accelerate using momentum conservation.
        D = reshape(D, (2 * n_x + 1, 2 * n_y + 1, 2 * n_x + 1, 2 * n_y + 1))
        D_padded = pad(D, ((0, 1), (0, 1), (0, 0), (0, 0)))
        D_shifted = D_padded[p2r_x_cut, p2r_y_cut, p2_x, p2_y]
        K_shifted = einsum("abcd,cdef->abef", v_kp1_flavors[i], D_shifted,
                           optimize="greedy")
        K = K_shifted[k1p_x, k1p_y, k1p_x - k1_x + 2 * n_x,
                      k1p_y - k1_y + 2 * n_y]
        K = reshape(K, (N_basis, N_basis))
        return (K)
    
    def fock_scratch(self, D_flavors: ndarray) -> ndarray:
        """
        Calculate the Fock matrix from scratch.

        Parameters
        ----------
        D_flavors : ndarray
            Density matrix for each flavor

        Returns
        -------
        F_flavors : ndarray
            Fock matrix for each flavor
        """

        N_flavors = self.N_flavors
        N_basis = self.N_basis
        T_flavors = self.T_flavors
        
        # Calculate Coulomb matrices.
        i, j = mgrid[0:N_flavors, 0:N_flavors].reshape(2, N_flavors ** 2)
        J_flavors: ndarray = self.coulomb(i, j, D_flavors[j, :, :])
        J_flavors = J_flavors.reshape(N_flavors, N_flavors, N_basis, N_basis)
        J_flavors = sum(J_flavors, axis=1)

        # Calculate exchange matrices.
        i = mgrid[0:N_flavors]
        K_flavors: ndarray = self.exchange(i, D_flavors[i, :, :])

        # Assemble, negative sign for exchange done here.
        F_flavors = T_flavors + J_flavors - K_flavors
        return (F_flavors)

    def fock(self, D_flavors: ndarray) -> ndarray:
        """
        Use a cache to avoid recomputing same argument back-to-back, which may
        occur because minimization uses both the function and its gradient.

        Parameters
        ----------
        D_flavors : ndarray
            Density matrix for each flavor

        Returns
        -------
        F_flavors : ndarray
            Fock matrix for each flavor
        """

        match = allclose(D_flavors, self.D_cache, rtol=1e-14, atol=1e-14)
        if match:
            return (self.F_cache)
        else:
            F_new = self.fock_scratch(D_flavors)
            self.D_cache = D_flavors.copy()
            self.F_cache = F_new.copy()
            self.fock_calls += 1
            return (F_new)
        

    """
    Functions for initialization.
    """
    def gaussian_orbital(self, x: float, y: float, a: float) -> ndarray:
        """
        Creates a Gaussian orbital

        Parameters
        ----------
        x : float
            x-coordinate of center.
        y : float
            y-coordinate of center.
        a : float
            Width.

        Returns
        -------
        phi_u : ndarray
            Gaussian's expansion into plane wave.
        """

        def gaussian_FT(q_x: float, q_y: float) -> complex:
            """
            Fourier transform of Gaussian orbital.
            """

            argument = -1j * (x * q_x + y * q_y) - \
                (a ** 2) * (q_x ** 2 + q_y ** 2)
            return (exp(argument))

        q_x, q_y = self.k_vecs.T
        phi_u = gaussian_FT(q_x, q_y)
        phi_u = phi_u / linalg.norm(phi_u)
        return (phi_u)

    def gaussian_orbitals(self, positions: ndarray, a: float) -> ndarray:
        """
        Creates Gaussian orbitals at several positions (but same width)

        Parameters
        ----------
        positions : ndarray
            rs[i, d] = (d) coordinate of particle (i).
        a : float
            Width of Gaussian.

        Returns
        -------
        C : ndarray
            C[u, i] = Component of Gaussian (i) on plane wave (u).
        """

        N_orbitals, _ = positions.shape
        C = zeros((self.N_basis, N_orbitals), dtype=complex128)
        for i in range(N_orbitals):
            x, y = positions[i, 0], positions[i, 1]
            C = C.at[:, i].set(self.gaussian_orbital(x, y, a))
        return (C)

        
    """
    Functions to calculate the energy and gradient and optimize.
    """
    def unflatten(self, C: ndarray) -> list[ndarray]:
        """
        Unflatten description vector into flavor coefficient matrices (SciPy
        optimizers require 1D real arrays).

        Parameters
        ----------
        C : ndarray
            Real vector describing HF ansatz.

        Returns
        -------
        C_flavors : list
            Coefficient matrix for each flavor.
        """

        N_basis = self.N_basis
        pop_flavors = self.pop_flavors

        # Convert back to complex.
        C_real = C[:len(C) // 2]
        C_imag = C[len(C) // 2:]
        C_complex = C_real + 1j * C_imag

        # Recover individual coefficient matrices for each flavor
        C_flavors = []
        start = 0 
        for pop in pop_flavors:
            num_coefficients = N_basis * pop
            C_flavor = C_complex[start:start + num_coefficients]
            C_flavor = reshape(C_flavor, (N_basis, pop))
            C_flavors.append(C_flavor)
            start += num_coefficients
        return (C_flavors)
    
    def energy_CG(self, C: ndarray) -> float:
        """
        Calculate the Hartree-Fock energy. 

        Parameters
        ----------
        C : ndarray
            Real vector describing HF ansatz.

        Returns
        -------
        E : float
            Total energy.
        """

        C_flavors = self.unflatten(C)
        _, D_flavors = S_inv_and_D_flavors(C_flavors)
        F_flavors = self.fock(D_flavors)
        E = sum((self.T_flavors + F_flavors) * D_flavors).real / 2
        self.E = E
        return (E)

    def gradient_CG(self, C: ndarray) -> ndarray:
        """
        Calculate the derivative of the analytically-continued Hartree-Fock
        energy. 

        Parameters
        ----------
        C : ndarray
            Real vector describing HF ansatz.

        Returns
        -------
        gradient_all : ndarray
            Gradient of energy as a real vector.
        """

        C_flavors = self.unflatten(C)
        S_inv_flavors, D_flavors = S_inv_and_D_flavors(C_flavors)
        F_flavors = self.fock(D_flavors)
        dE_dC_flavors = []
        for C, S_inv, F in zip(C_flavors, S_inv_flavors, F_flavors):
            dE_dC_flavors.append(gradient(C, S_inv, F))
        dE_dC = flatten(dE_dC_flavors)
        return (dE_dC)

    def run(self, C_initial = None, minimizer: dict = {"gtol": 1e-7, "norm": 2}):
        """
        Minimizes the energy of the Hartree-Fock state.

        Parameters
        ----------
        C_initial: list or None
            List of initial coefficient matrices for each flavor or None to
            indicate radom initialization.
        minimizer: dict, default = {"gtol": 1e-7, "norm": 2}
            Additional kwargs for scipy.fmin_cg.

        Sets
        ----
        fock_calls: int
            Number of from-scratch Fock matrix evaluations.
        cg_iters: int
            Number of conjugate gradient iterations.
        C_log: list
            List of HF state (flattened) at every CG iteration.
        E_log: List
            Total evergy at every CG iteration.
        result: OptimizeResult
            Output of CG algorithm.
        C_final: list
            Coefficient matrix for each flavor (NOT FLATTENED!).
        """

        N_flavors = self.N_flavors
        N_basis = self.N_basis
        pop_flavors = self.pop_flavors
        lowest_pop_flavors = self.lowest_pop_flavors

        # Compute energy of noninteracting Fermi sea for reference.
        C_free = []
        for i in range(N_flavors):
            C_i = zeros((N_basis, pop_flavors[i]), dtype=prec)
            C_i = C_i.at[lowest_pop_flavors[i], arange(pop_flavors[i])].set(1)
            C_free.append(C_i)
        C_free = flatten(C_free)
        E_free = self.energy_CG(C_free)
        print(f"Energy of free Fermi sea: {E_free}.\n")

        # Use random initialization.
        if C_initial is None:
            C_flavors_initial = []
            for i, pop in enumerate(pop_flavors):
                C_i = array(random.normal(size=(N_basis, pop)), prec)
                C_flavors_initial.append(lowdin(C_i))
            C_initial: ndarray = flatten(C_flavors_initial)
        else:
            C_initial = flatten(C_initial)

        # Logs to monitor convergence or troubleshoot.
        self.fock_calls = 0
        self.cg_iters = 0
        self.C_log = []
        self.E_log = []

        # Fancy header.
        template = "{: <9} | {: <10} | {: <20} | {: <20} | {: <5}"
        def helper(C: ndarray) -> None:
            """
            Report and log progress during minimization.

            Parameters
            ----------
            C : ndarray
                Current HF state.
            """

            self.cg_iters += 1
            self.C_log.append(C)
            self.E_log.append(self.E)
            time = perf_counter() - t_start
            if self.cg_iters == 1:
                delta_E = ""
            else:
                delta_E = log10(self.E_log[-2] - self.E_log[-1])
            data = [self.cg_iters, self.fock_calls, self.E, delta_E,
                    int(round(time))]
            print(template.format(*data))
            
        # Minimize with conjugate gradient.
        label = template.format(*["Iteration", "Fock Calls", "Energy", 
                                  "Delta Energy (Log10)", "Time"])
        print(label + "\n" + "_" * len(label))
        t_start = perf_counter()
        result = optimize.fmin_cg(self.energy_CG, C_initial, self.gradient_CG,
                                  disp=1, callback=helper, **minimizer)
        self.result = result
        self.C_final = [lowdin(C_i) for C_i in self.unflatten(result)]


    """
    Functions for post-processing, visualization, and saving.
    """
    def check_basis(self):
        """
        Calculates and plots the occupancy of each plane wave basis state
        to check convegence.
        """

        n_x = self.n_x
        n_y = self.n_y
        L_x = self.L_x
        L_y = self.L_y
        C_final = self.C_final
        N_flavors = self.N_flavors
        C_flavors = [lowdin(C_i) for C_i in C_final]

        # Calculate occupancy of each plane-wave state.
        _, D_flavors = S_inv_and_D_flavors(C_flavors)
        n_k_flavors = real(diagonal(D_flavors, axis1=1, axis2=2))
        n_k_flavors = n_k_flavors.reshape(N_flavors, 2 * n_x + 1, 2 * n_y + 1)

        # Plot Log10 of occupancies.
        aspect = (n_x / L_x) / (n_y / L_y) * (2 * n_x + 1) / (2 * n_y + 1)
        fig, axs = plt.subplots(1, N_flavors, figsize=(N_flavors * 3, 3))
        for i, n_k in enumerate(n_k_flavors):
            if N_flavors == 1:
                ax = axs
            else:
                ax: plt.Axes = axs[i]
            im = ax.imshow(log10(n_k).T, vmin=-10, vmax=0, cmap="magma",
                           origin="lower", aspect=aspect)
            ax.set_title(self.flavors[i])
            ax.set_xticks([], [])
            ax.set_yticks([], [])

        # Append colorbar to right side.
        corners = ax.get_position().get_points()
        height = corners[1][1] - corners[0][1]
        width = (corners[1][0] - corners[0][0]) / 10
        left = corners[1][0] + width
        bottom = corners[0][1]
        cbar_ax: plt.Axes = fig.add_axes([left, bottom, width, height])
        plt.colorbar(im, cax=cbar_ax)
        cbar_ax.set_yticks([-10, -5, 0])

        # Log.
        self.n_k_flavors = n_k_flavors

    def visualizer_setup(self, pixels: int):
        """
        Setup the visualization grid and the plane-wave basis's values in
        real-space.

        Parameters
        ----------
        pixels : int
            Approximate total number of pixels in image.
        """

        L_x = self.L_x
        L_y = self.L_y

        P_x = int(sqrt(pixels * L_x / L_y)) # Pixels in x-direction.
        P_y = int(sqrt(pixels * L_y / L_x)) # Pixels in y-direction.
        r_grid = mgrid[0:P_x, 0:P_y].reshape(2, P_x * P_y).T *\
                array([L_x / P_x, L_y / P_y])
        phi_ur = (1 / sqrt(L_x * L_y)) * exp(1j * self.k_vecs @ r_grid.T)
        
        self.P_x = P_x
        self.P_y = P_y
        self.phi_ur = phi_ur # Plane-wave (u)'s value at position (r).
    
    def calc_rho(self, C: ndarray) -> ndarray:
        """
        Calculate number density.

        Parameters
        ----------
        C : ndarray
            ON coefficient matrix.

        Returns
        -------
        rho_r : ndarray
            Number density corresponding to the Lowdin orthogonalized
            coefficients on the visualization grid.
        """
        phi_ur = self.phi_ur
        P_x = self.P_x
        P_y = self.P_y

        phi_ir = C.T @ phi_ur
        rho_r = sum(abs(phi_ir) ** 2, axis=0)
        rho_r = reshape(rho_r, (P_x, P_y)).real.T
        return (rho_r)