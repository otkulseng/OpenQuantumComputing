import numpy as np
import math

from ansatz import DefaultAnsatz
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter

class ExactCover(DefaultAnsatz):
    def __init__(self, params={}) -> None:
        super().__init__()

        self.FR = params.get("FR", None)
        self.CR = params.get("CR", None)
        self.mu = params.get("mu", 1)
        self.N_qubits = params.get("instances")


    def _exact_cover(self, x):
        return np.sum((1 - (self.FR @ x)) ** 2)

    def create_phase(self):
        """
        Creates parameterized circuit corresponding to the cost function
        """
        q = QuantumRegister(self.N_qubits)
        self.cost_circuit = QuantumCircuit(q)
        cost_param = Parameter("x_gamma")
        if self.usebarrier:
            self.parameterized_circuit.barrier()

        F, R = np.shape(self.FR)

        ### cost Hamiltonian
        for r in range(R):
            hr = self.mu * 0.5 * self.FR[:, r] @ (np.sum(self.FR, axis=1) - 2)
            if not self.CR is None:
                hr += 0.5 * self.CR[r]

            if not math.isclose(hr, 0, abs_tol=1e-7):
                self.cost_circuit.rz(cost_param * hr, q[r])

            for r_ in range(r + 1, R):
                Jrr_ = self.mu * 0.5 * self.FR[:, r] @ self.FR[:, r_]

                if not math.isclose(Jrr_, 0, abs_tol=1e-7):
                    self.cost_circuit.cx(q[r], q[r_])
                    self.cost_circuit.rz(cost_param * Jrr_, q[r_])
                    self.cost_circuit.cx(q[r], q[r_])
        if self.usebarrier:
            self.cost_circuit.barrier()


    def cost(self, string):
        x = np.array(list(map(int, string)))
        c_e = self._exactCover(x)

        if self.CR is None:
            return -c_e
        else:
            return -(self.CR @ x + self.mu * c_e)

    def isFeasible(self, string, feasibleOnly=False):
        x = np.array(list(map(int, string)))
        c_e = self.__exactCover(x)
        if math.isclose(c_e, 0, abs_tol=1e-7):
            return True
        else:
            return False

