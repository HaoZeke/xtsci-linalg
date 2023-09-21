import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


def test_conjugate_gradient_nocond():
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    x0 = np.array([2, 1])  # Initial guess

    maxIters = 1000
    tol = 1e-5
    solution, _ = cg(A, b, x0=x0, maxiter=maxIters, tol=tol)

    expected_solution = np.array(
        [0.09090909090909093936971885341336019, 0.63636363636363624252112458634655923]
    )
    print(f"Computed solution: [{solution[0]:.15f}, {solution[1]:.15f}]")
    print(
        f"Expected solution: [{expected_solution[0]:.15f}, {expected_solution[1]:.15f}]"
    )

    # Check for convergence
    if not np.allclose(solution, expected_solution, atol=1e-4):
        print("The solutions are not sufficiently close!")
    else:
        print("Converged!")


class JacobiPreconditioner:
    def __init__(self, A):
        self.M_inv = 1.0 / np.diag(A)

    def __call__(self, x):
        return self.M_inv * x

    def matvec(self, x):
        return self.M_inv * x

    def rmatvec(self, x):
        return self.M_inv * x


def test_conjugate_gradient():
    A = np.array([[4, 1], [1, 3]])
    b = np.array([1, 2])
    x0 = np.array([2, 1])  # Initial guess

    maxIters = 1000
    tol = 1e-5

    M = JacobiPreconditioner(A)
    M_op = LinearOperator(A.shape, M.matvec, M.rmatvec, dtype=A.dtype)

    solution, _ = cg(A, b, x0=x0, maxiter=maxIters, tol=tol, M=M_op)

    expected_solution = np.array([0.0909, 0.6364])
    print(f"Computed solution: [{solution[0]:.35f}, {solution[1]:.35f}]")
    print(
        f"Expected solution: [{expected_solution[0]:.15f}, {expected_solution[1]:.15f}]"
    )

    # Check for convergence
    if not np.allclose(solution, expected_solution, atol=1e-4):
        print("The solutions are not sufficiently close!")
    else:
        print("Converged!")


if __name__ == "__main__":
    test_conjugate_gradient()
