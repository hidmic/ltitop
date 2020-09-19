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


class SpecializedImplicitModel(Model):

    @dataclass(frozen=True)
    class Parameters:
        __slots__ = ('J', 'M', 'N', 'K', 'P', 'Q', 'L', 'R', 'S')

        def __new__(cls, J, M, N, K, P, Q, L, R, S):
            n_t, n_x, n_u, n_y = J.rows, K.rows, N.cols, L.rows
            if J.shape != (n_t, n_t):
                raise ValueError()
            if M.shape != (n_t, n_x):
                raise ValueError()
            if N.shape != (n_t, n_u):
                raise ValueError()
            if K.shape != (n_x, n_t):
                raise ValueError()
            if P.shape != (n_x, n_x):
                raise ValueError()
            if Q.shape != (n_x, n_u):
                raise ValueError()
            if L.shape != (n_y, n_t):
                raise ValueError()
            if R.shape != (n_y, n_x):
                raise ValueError()
            if S.shape != (n_y, n_u):
                raise ValueError()
            return super().__new__(cls, J, M, N, K, P, Q, L, R, S)

        @memoize
        def to_state_space(self):
            J_inverse = self.J.inv()
            return StateSpaceRealization.Parameters(
                A=self.K * J_inverse * self.M + self.P,
                B=self.K * J_inverse * self.N + self.Q,
                C=self.L * J_inverse * self.M + self.R,
                D=self.L * J_inverse * self.N + self.S
            )

        @memoize
        def to_matrix(self):
            return sympy.Matrix([
                [-self.J, self.M, self.N],
                [ self.K, self.P, self.Q],
                [ self.L, self.R, self.S]
            ])

    @staticmethod
    def _make_algorithm(n_t, n_x, n_u, n_y):
        # Define inputs (parameters included)
        X_k = sympy.MatrixSymbol('X(k)', n_x, 1)
        U_k = sympy.MatrixSymbol('U(k)', n_u, 1)
        J = sympy.MatrixSymbol('J', n_t, n_t)
        M = sympy.MatrixSymbol('M', n_t, n_x)
        N = sympy.MatrixSymbol('M', n_t, n_u)
        K = sympy.MatrixSymbol('K', n_x, n_t)
        P = sympy.MatrixSymbol('P', n_x, n_x)
        Q = sympy.MatrixSymbol('Q', n_x, n_u)
        L = sympy.MatrixSymbol('K', n_y, n_t)
        R = sympy.MatrixSymbol('P', n_y, n_x)
        S = sympy.MatrixSymbol('Q', n_y, n_u)
        inputs = X_k, U_k, J, M, N, K, P, Q, L, R, S

        # Define outputs
        T_kk = sympy.MatrixSymbol('T(k + 1)', n_t, 1)
        X_kk = sympy.MatrixSymbol('X(k + 1)', n_x, 1)
        Y_k = sympy.MatrixSymbol('Y(k)', n_y, 1)
        outputs = T_kk, X_kk, Y_k

        # Define algorithm
        mT_kk = sympy.Matrix(T_kk)
        mX_kk = sympy.Matrix(X_kk)
        mY_k = sympy.Matrix(Y_k)
        mX_k = sympy.Matrix(X_k)
        mU_k = sympy.Matrix(U_k)
        mJ = sympy.Matrix(J)
        mM = sympy.Matrix(M)
        mN = sympy.Matrix(M)
        mK = sympy.Matrix(K)
        mP = sympy.Matrix(P)
        mQ = sympy.Matrix(Q)
        mL = sympy.Matrix(K)
        mR = sympy.Matrix(P)
        mS = sympy.Matrix(Q)

        subalgorithms = []

        # Assume J is triangular
        subalgorithms.append(tuple(
            [Assignment(mT_kk[0], nonassociative(
                mM[0, :] * mX_k + mN[0, :] * mU_k
            ))] +
            [Assignment(mT_kk[i], nonassociative(
                -mJ[i, :i] * mT_kk[:i] +
                mM[i, :] * mX_k +
                mN[i, :] * mU_k
            )) for i in range(1, n_t)] +
            [Assignment(T_kk, mT_kk)]
        ))

        subalgorithms.append(tuple(
            Assignment(mX_kk[i], nonassociative(
                mK[i, :] * mT_kk +
                mP[i, :] * mX_k +
                mQ[i, :] * mU_k
            )) for i in range(n_x)
        ))

        subalgorithms.append(tuple(
            Assignment(mY_k[i], nonassociative(
                mL[i, :] * mT_kk +
                mR[i, :] * mX_k +
                mS[i, :] * mU_k
            )) for i in range(n_y)
        ))

        subalgorithms = tuple(subalgorithms)

        return Algorithm(inputs, outputs, subalgorithms)

    def __new__(cls, J, M, N, K, P, Q, L, R, S):
        parameters = cls.Parameters(
            J, M, N, K, P, Q, L, R, S
        )
        algorithm = cls._make_algorithm(
            parameters.n_t, parameters.n_x,
            parameters.n_u, parameters.n_y
        )
        return super().__new__(cls, parameters, algorithm)

    @property
    def n_t(self):
        return self.model.J.rows

    @property
    def n_x(self):
        return self.params.K.rows

    @property
    def n_u(self):
        return self.params.N.cols

    @property
    def n_y(self):
        return self.params.L.rows

    def similarity_transform(self, Y, U, W):
        U_inverse = U.inv()
        J, M, N, K, P, Q, L, R, S = self.parameters
        return type(self)(
            J=Y * J * W, M=Y * M * U, N=Y * N,
            K=U_inverse * K * W, P=U_inverse * P * U,
            Q=U_inverse * Q, L=L * W, R=R * U, S=S,
        )

    @memoize
    def to_state_space(self):
        return StateSpaceModel(*self.parameters.to_state_space())

    def process(self, U, X=None):
        if X is None:
            X = sympy.zeros(self.n_x, 1)
        J, M, N, K, P, Q, L, R, S = astuple(self.parameters)

        return self.algorithm.perform(
            X[k], U[k], J, M, N, K, P, Q, L, R, S
        )
        for k in range(len(U)):
            X[k + 1], Y[k] = self.algorithm.perform(
                X[k], U[k], J, M, N, K, P, Q, L, R, S
            )
        return X, Y

