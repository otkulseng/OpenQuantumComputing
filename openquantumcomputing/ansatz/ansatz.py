import structlog
LOG = structlog.get_logger(file=__name__)

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from abc import ABC, abstractmethod

class Ansatz(ABC):

    @abstractmethod
    def set_initial_state(self, qubits):
        pass

    @abstractmethod
    def create_phase(self):
        pass

    @abstractmethod
    def create_mixer(self):
        pass

    @abstractmethod
    def cost(self, string):
        pass


class DefaultAnsatz(Ansatz):
    def create_mixer(self):
        """
        Implements the function that initializes the member variable
        self.mixer_circuit as a parameterized circuit

        Overwritten in child classes where a constraint preserving mixer is used, for example the XY-mixer
        """

        q = QuantumRegister(self.N_qubits)
        mixer_param = Parameter("x_beta")

        self.mixer_circuit = QuantumCircuit(q)
        self.mixer_circuit.rx(-2 * mixer_param, range(self.N_qubits))

        if self.usebarrier:
            self.mixer_circuit.barrier()



