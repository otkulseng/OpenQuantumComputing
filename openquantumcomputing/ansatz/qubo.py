import numpy as np
import math

from ansatz import Ansatz


class QUBO(Ansatz):
    def __init__(self) -> None:
        """
        init function that initializes QUBO.
        The aim is to solve the problem
        min x^T Q x + c^T x + b
        for n-dimensional binary variable x

        :param params: additional parameters
        """
        super().__init__()

        self.parameterized = False
        self.QUBO_Q = None
        self.QUBO_c = None
        self.QUBO_b = None
        self.lower_triangular_Q = False

    def cost(self, string):
        x = np.array(list(map(int, string)))
        return -(x.T @ self.QUBO_Q @ x + self.QUBO_c.T @ x + self.QUBO_b)

    def _init_QUBO(self, Q=None, c=None, b=None):
        """
        Implements the mapping from the parameters in params to the QUBO problem.
        Is expected to be called by the child class.
        """
        assert type(Q) is np.ndarray, "Q needs to be a numpy ndarray, but is " + str(
            type(Q)
        )
        assert (
            Q.ndim == 2
        ), "Q needs to be a 2-dimensional numpy ndarray, but has dim " + str(Q.ndim)
        assert Q.shape[0] == Q.shape[1], "Q needs to be a square matrix, but is " + str(
            Q.shape
        )
        n = Q.shape[0]

        # Check if Q is lower triangular
        self.lower_triangular_Q = np.allclose(Q, np.tril(Q))

        self.QUBO_Q = Q

        if c is None:
            c = np.zeros(n)
        assert type(c) is np.ndarray, "c needs to be a numpy ndarray, but is " + str(
            type(c)
        )
        assert (
            c.ndim == 1
        ), "c needs to be a 1-dimensional numpy ndarray, but has dim " + str(Q.ndim)
        assert c.shape[0] == n, (
            "c is of size "
            + str(c.shape[0])
            + " but should be compatible size to Q, meaning "
            + str(n)
        )
        self.QUBO_c = c

        if b is None:
            b = 0.0
        assert np.isscalar(b), "b is expected to be scalar, but is " + str(b)
        self.QUBO_b = b

class Portfolio(QUBO):
    def __init__(self, params) -> None:
        super().__init__()

        self.__checkParams(params)
        self.risk = params["risk"]
        self.budget = params["budget"]
        self.cov_matrix = params["cov_matrix"]
        self.exp_return = params["exp_return"]
        self.penalty = params.get("penalty", 0.0)

        # Reformulated as a QUBO
        # min x^T Q x + c^T x + b
        # Writing Q as lower triangular matrix since it otherwise is symmetric
        # Q = self.risk * np.tril(self.cov_matrix + np.tril(self.cov_matrix, k=-1)) \
        #               + self.penalty*(np.eye(self.N_assets) + 2* np.tril(np.ones((self.N_assets, self.N_assets)), k=-1))
        # c = - self.exp_return - (2*self.penalty*self.budget*np.ones_like(self.exp_return))
        # b = self.penalty*self.budget*self.budget

        Q = self.risk * np.tril(
            self.cov_matrix + np.tril(self.cov_matrix, k=-1)
        ) + self.penalty * (
            np.eye(self.N_qubits)
            + 2 * np.tril(np.ones((self.N_qubits, self.N_qubits)), k=-1)
        )
        c = -self.exp_return - (
            2 * self.penalty * self.budget * np.ones_like(self.exp_return)
        )
        b = self.penalty * self.budget * self.budget

        self._init_QUBO(Q=Q, c=c, b=b)


    def __checkParams(self, params):
        # we require the following params:
        for key in ["risk", "budget", "cov_matrix", "exp_return"]:
            assert key in params, "missing required parameter " + key

    def __str2np(self, s):
        x = np.array(list(map(int, s)))
        assert len(x) == len(self.params.get("exp_return")), (
            "bitstring  "
            + s
            + " of wrong size. Expected "
            + str(len(self.params.get("exp_return")))
            + " but got "
            + str(len(x))
        )
        return x

    def cost_nonQUBO(self, string, penalize=True):
        x = np.array(list(map(int, string)))
        cost = self.risk * (x.T @ self.cov_matrix @ x) - self.exp_return.T @ x
        if penalize:
            cost += self.penalty * (x.sum() - self.budget) ** 2

        return -cost

    def isFeasible(self, string, feasibleOnly=False):
        x = self.__str2np(string)
        constraint = np.sum(x) - self.params.get("budget")
        if math.isclose(constraint, 0, abs_tol=1e-7):
            return True
        else:
            return False
