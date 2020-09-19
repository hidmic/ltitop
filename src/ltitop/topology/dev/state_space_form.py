# -*- coding: utf-8 -*-

# ltitop - A toolkit to describe and optimize LTI systems topology
# Copyright (C) 2021 Michel Hidalgo <hid.michel@gmail.com>
#
# This file is part of ltitop.
#
# ltitop is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ltitop is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with ltitop.  If not, see <http://www.gnu.org/licenses/>.


import itertools
import collections
import sympy
import numpy


@immutable_dataclass
class StateSpaceErrorModel(Model):

    @immutable_dataclass
    class Parameters:
        A: sympy.Matrix
        Aq: sympy.Matrix
        dA: sympy.Matrix
        B: sympy.Matrix
        dB: sympy.Matrix
        Cq: sympy.Matrix
        dC: sympy.Matrix
        dD: sympy.Matrix

        def __post_init__(self):

    @staticmethod
    def _make_algorithm(n_x, n_u, n_y):
        # Define inputs (parameters included)
        X_k = sympy.MatrixSymbol('X(k)', n_x, 1)
        dX_k = sympy.MatrixSymbol('∆X(k)', n_x, 1)
        U_k = sympy.MatrixSymbol('U(k)', n_u, 1)
        Ex_k = sympy.MatrixSymbol('Ɛx(k)', n_x, 1)
        Ey_k = sympy.MatrixSymbol('Ɛy(k)', n_y, 1)
        A = sympy.MatrixSymbol('A', n_x, n_x)
        Aq = sympy.MatrixSymbol('Ā', n_x, n_x)
        dA = sympy.MatrixSymbol('∆A', n_x, n_x)
        B = sympy.MatrixSymbol('B', n_x, n_u)
        dB = sympy.MatrixSymbol('∆B', n_x, n_u)
        Cq = sympy.MatrixSymbol('C̄', n_y, n_x)
        dC = sympy.MatrixSymbol('∆C', n_y, n_x)
        dD = sympy.MatrixSymbol('∆D', n_y, n_u)
        inputs = X_k, dX_k, U_k, Ex_k, Ey_k, A, Aq, dA, B, dB, Cq, dC, dD

        # Define outputs
        X_kk = sympy.MatrixSymbol('X(k + 1)', n_x, 1)
        dX_kk = sympy.MatrixSymbol('∆X(k + 1)', n_x, 1)
        dY_k = sympy.MatrixSymbol('∆Y(k)', n_y, 1)
        outputs = X_kk, dX_kk, dY_k

        # Define algorithm
        mX_kk = sympy.Matrix(X_kk)
        mdX_kk = sympy.Matrix(dX_kk)
        mdY_k = sympy.Matrix(dY_k)
        mX_k = sympy.Matrix(X_k)
        mdX_k = sympy.Matrix(dX_k)
        mU_k = sympy.Matrix(U_k)
        mEx_k = sympy.Matrix(Ex_k)
        mEy_k = sympy.Matrix(Ey_k)
        mA = sympy.Matrix(A)
        mAq = sympy.Matrix(Aq)
        mdA = sympy.Matrix(dA)
        mB = sympy.Matrix(B)
        mdB = sympy.Matrix(dB)
        mCq = sympy.Matrix(Cq)
        mdC = sympy.Matrix(dC)
        mdD = sympy.Matrix(dD)

        subalgorithms = []

        subalgorithms.append(tuple(
            Assignment(mX_kk[i], nonassociative(
                mA[i, :] * mX_k + mB[i, :] * mU_k
            )) for range i in range(n_x)
        ) + (Assignment(X_kk, mX_kk),))

        subalgorithms.append(tuple(
            Assignment(mdX_kk[i], nonassociative(
                mdA[i, :] * mX_k + mAq[i, :] * mdX_k +
                mdB[i, :] * mU_k + mEx_k[i]
            )) for range i in range(n_x)
        ) + (Assignment(dX_kk, mdX_kk),))

        subalgorithms.append(tuple(
            Assignment(mdY_k[i], nonassociative(
                mdC[i, :] * mX_k + mCq[i, :] * mdX_k +
                mdD[i, :] * mU_k + mEy_k[i]
            )) for range i in range(n_y)
        ) + (Assignment(dY_k, mdY_k),))

        subalgorithms = tuple(subalgorithms)

        return Algorithm(inputs, outputs, subalgorithms)

    def __init__(cls, A, B, C, D, quantizer=fixed):
        algorithm = cls._make_algorithm(A.rows, B.cols, C.rows)
        return super().__new__(cls, parameters, algorithm)

    @memoize
    def to_state_space(self):
        Ix = sympy.eye(self.n_x)
        Iy = sympy.eye(self.n_y)
        Ox = sympy.zeros(self.n_x)
        A, Aq, dA, B, dB, Cq, dC, dD = self.parameters
        return StateSpaceModel(
            A=sympy.Matrix.vstack(
                sympy.Matrix.hstack(A, Ox),
                sympy.Matrix.hstack(dA, Aq)
            ),
            B=sympy.Matrix.vstack(
                sympy.Matrix.hstack(B, Ox),
                sympy.Matrix.hstack(dB, Ix)
            ),
            C=sympy.Matrix.hstack(dC, Cq),
            D=sympy.Matrix.hstack(dD, Iy)
        )


@dataclass
class StateSpaceModel(Model):

    @dataclass
    class Parameters:
        A: sympy.Matrix
        B: sympy.Matrix
        C: sympy.Matrix
        D: sympy.Matrix

        def __post_init__(self):
            n_x = self.A.rows
            if A.shape != (n_x, n_x):
                raise ValueError()
            n_u = self.B.cols
            if B.shape != (n_x, n_u):
                raise ValueError()
            n_y = self.C.rows
            if C.shape != (n_y, n_x):
                raise ValueError()
            if D.shape != (n_y, n_u):
                raise ValueError()

        @memoize
        def to_matrix(self):
            return sympy.Matrix([
                [self.A, self.B],
                [self.C, self.D]
            ])

    @staticmethod
    @memoize
    def _make_algorithm(n_x, n_u, n_y):
        # Define inputs (parameters included)
        X_k = sympy.MatrixSymbol('X(k)', n_x, 1)
        U_k = sympy.MatrixSymbol('U(k)', n_u, 1)
        A = sympy.MatrixSymbol('A', n_x, n_x)
        B = sympy.MatrixSymbol('B', n_x, n_u)
        C = sympy.MatrixSymbol('C', n_y, n_x)
        D = sympy.MatrixSymbol('D', n_y, n_u)
        inputs = X_k, U_k, A, B, C, D

        # Define outputs
        X_kk = sympy.MatrixSymbol('X(k + 1)', n_x, 1)
        Y_k = sympy.MatrixSymbol('Y(k)', n_y, 1)
        outputs = X_kk, Y_k

        # Define algorithm
        mX_kk = sympy.Matrix(X_kk)
        mX_k = sympy.Matrix(X_k)
        mU_k = sympy.Matrix(U_k)
        mY_k = sympy.Matrix(Y_k)
        mA = sympy.Matrix(A)
        mB = sympy.Matrix(B)
        mC = sympy.Matrix(C)
        mD = sympy.Matrix(D)

        procedures = []
        procedures.append(tuple(
            Assignment(mX_kk[i], nonassociative(
                mA[i, :] * mX_k + mB[i, :] * mU_k
            )) for i in range(n_x)
        ) + (Assignment(X_kk, mX_kk),))
        procedures.append(tuple(
            Assignment(mY_k[i], nonassociative(
                mC[i, :] * mX_k + mD[i, :] * mU_k
            )) for i in range(n_y)
        ) + (Assignment(Y_k, mY_k),))
        procedures = tuple(procedures)

        return Algorithm(inputs, outputs, procedures)

    def __init__(self, A, B, C, D):
        parameters = StateSpaceModel.Parameters(A, B, C, D)
        algorithm = StateSpaceModel._make_algorithm(A.rows, B.cols, C.rows)
        return super().__init__(parameters, algorithm)

    @property
    def n_x(self):
        return self.parameters.A.rows

    @property
    def n_u(self):
        return self.parameters.B.cols

    @property
    def n_y(self):
        return self.parameters.C.rows

    @property
    @memoize
    def stable(self):
        return all(value < 0 for value in self.parameters.A.eigenvals())

    @property
    @memoize
    def dc_gain(self):
        A, B, C, D = self.parameters
        I = sympy.eye(self.n_x, self.n_x)
        return C * (I - A).inv() * B + D

    @property
    @memoize
    def worst_case_peak_gain(self):
        A, B, C, D = self.parameters
        return sympy.Matrix(WCPG_ABCD(
            numpy.matrix(A), numpy.matrix(B),
            numpy.matrix(C), numpy.matrix(D)
        ))

    wcpg = worst_case_peak_gain

    def similarity_transform(self, T):
        A, B, C, D = astuple(self.parameters)
        if A.shape != T.shape:
            raise ValueError()
        T_inverse = T.inv()
        return type(self)(
            parameters=type(self.parameters)(
                A=T * A * T_inverse, B=T * B,
                C=C * T_inverse, D=D
            ),
            algorithm=self.algorithm
        )

    def process(self, U, X0):
        Y = [None] * len(U)
        X = [None] * len(U)
        if X0 is None:
            X0 = sympy.zeros(self.n_x, 1)
        X[0] = X0
        A, B, C, D = self.parameters
        for k in range(len(U)):
            X[k + 1], Y[k] = \
                self.algorithm.perform(
                    X[k], U[k], A, B, C, D)
        return X, Y

