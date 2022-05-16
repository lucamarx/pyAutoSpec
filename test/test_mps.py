import unittest
import numpy as np

from pyautospec import Mps
from pyautospec.dmrg_learning import ContractionCache, _contract_left, _split, _merge, _gradient


class TestMps(unittest.TestCase):

    def is_left_canonical(self, mps, n):
        "Check that MPS is in left canonical form from start to site n"
        if not np.allclose(np.einsum("pj,pk->jk", mps[0], mps[0]), np.eye(mps[0].shape[1])):
            return False

        for n in range(1, n+1):
            if not np.allclose(np.einsum("ipj,ipk->jk", mps[n], mps[n]), np.eye(mps[n].shape[2])):
                raise False

        return True


    def is_right_canonical(self, mps, n):
        "Check that MPS is in right canonical form from site n to end"
        if not np.allclose(np.einsum("ip,jp->ij", mps[mps.N-1], mps[mps.N-1]), np.eye(mps[mps.N-1].shape[0])):
            return False

        for n in reversed(range(n,mps.N-1)):
            if not np.allclose(np.einsum("ipk,jpk->ij", mps[n], mps[n]), np.eye(mps[n].shape[0])):
                return False

        return True


    def test_creation(self):
        mps = Mps(N=64, part_d=2)

        self.assertEqual(mps.N, 64)
        self.assertEqual(mps.part_d, 2)

        X = np.random.rand(16, 64, 2)

        self.assertEqual(mps(X).shape, (16,),
                         msg="mps evaluated over a batch must be a vector")


    # def test_canonical(self):
    #     mps = Mps(N=64, part_d=2)

    #     self.assertTrue(self.is_left_canonical(mps, mps.N-2),
    #                     msg="newly created mps must be left canonical")


    def test_contraction_cache(self):
        mps = Mps(N=64, part_d=2)

        X = np.random.rand(32, 64, 2)

        cache = ContractionCache(mps, X)

        for n in range(62):
            c = _contract_left(mps, n, X)

            self.assertEqual(c.shape, (32, 2),
                             msg="left contraction must produce a batch of vectors")

            self.assertTrue(np.allclose(cache[n], c),
                            msg="left contraction must agree with cache")


    def test_merge(self):
        mps = Mps(N=64, part_d=3)

        B_head = _merge(mps, 0)
        self.assertEqual(B_head.shape, (3,3,2),
                         msg="head bond tensor must be rank 3")

        B_tail = _merge(mps, 62)
        self.assertEqual(B_tail.shape, (2,3,3),
                         msg="tail bond tensor must be rank 3")

        for n in range(1,62):
            B_inner = _merge(mps, n)
            self.assertEqual(B_inner.shape, (2,3,3,2),
                             msg="inner bond tensors must be rank 4")


    def test_split(self):
        p = 5
        mps = Mps(N=64, part_d=p)

        for l in [True, False]:
            B_head = _merge(mps, 0)

            A1_head, A2_head = _split(mps, 0, B_head, left=l)

            d_upd = A1_head.shape[1]
            self.assertEqual(A1_head.shape, (p,d_upd),
                             msg="head tensor must have shape (p,d)")

            self.assertEqual(A2_head.shape, (d_upd,p,2),
                             msg="any tensor must have shape (d,p,d)")

            self.assertTrue(np.allclose(B_head, np.einsum("pi,iqj->pqj", A1_head, A2_head)),
                            msg="tensor contraction must give head bond tensor back")

            B_tail = _merge(mps, 62)

            A1_tail, A2_tail = _split(mps, 62, B_tail, left=l)

            d_upd = A1_tail.shape[1]
            self.assertEqual(A1_tail.shape, (2,p,d_upd),
                             msg="any tensor must have shape (d,p,d)")

            self.assertEqual(A2_tail.shape, (d_upd,p),
                             msg="tail tensor must have shape (d,p)")

            self.assertTrue(np.allclose(B_tail, np.einsum("ipj,jq->ipq", A1_tail, A2_tail)),
                            msg="tensor contraction must give tail bond tensor back")

            for n in range(1,62):
                B_inner = _merge(mps, n)

                A1_inner, A2_inner = _split(mps, n, B_inner, left=l)

                d_upd = A1_inner.shape[2]
                self.assertEqual(A1_inner.shape, (2,p,d_upd),
                                 msg="any tensor must have shape (d,p,d)")

                self.assertEqual(A2_inner.shape, (d_upd,p,2),
                                 msg="any tensor must have shape (d,p,d)")

                self.assertTrue(np.allclose(B_inner, np.einsum("ipj,jqk->ipqk", A1_inner, A2_inner)),
                                msg="tensor contraction must give bond tensor back")


    def test_split_canonical(self):
        p = 5
        mps = Mps(N=64, part_d=p)

        for n in range(1,62):
            B = _merge(mps, n)

            A1_left, A2_left = _split(mps, n, B, left=True)

            self.assertTrue(np.allclose(B, np.einsum("ipj,jqk->ipqk", A1_left, A2_left)),
                            msg="tensor contraction must give bond tensor back (left)")

            self.assertTrue(np.allclose(np.einsum("ipj,ipk->jk", A2_left, A2_left), np.eye(A2_left.shape[2])),
                            msg="left split must leave mps left canonical")

            A1_right, A2_right = _split(mps, n, B, left=False)

            self.assertTrue(np.allclose(B, np.einsum("ipj,jqk->ipqk", A1_right, A2_right)),
                            msg="tensor contraction must give bond tensor back (right)")

            self.assertTrue(np.allclose(np.einsum("ipk,jpk->ij", A2_right, A2_right), np.eye(A2_right.shape[0])),
                            msg="right split must leave mps right canonical")


    def test_gradient(self):
        p = 5
        mps = Mps(N=64, part_d=p)

        X = np.random.rand(16, 64, p)
        y = np.random.rand(16)

        B_head = _merge(mps, 0)

        G_head = _gradient(mps, 0, X, y)

        self.assertEqual(G_head.shape, B_head.shape,
                         msg="gradient shape must agree with bond tensor (head)")

        B_tail = _merge(mps, 62)

        G_tail = _gradient(mps, 62, X, y)

        self.assertEqual(G_tail.shape, B_tail.shape,
                         msg="gradient shape must agree with bond tensor (tail)")

        for n in range(1,62):
            B_inner = _merge(mps, n)

            G_inner = _gradient(mps, n, X, y)

            self.assertEqual(G_inner.shape, B_inner.shape,
                             msg="gradient shape must agree with bond tensor")


    # def test_sweep(self):
    #     p = 5
    #     mps = Mps(N=64, part_d=p)

    #     X = np.random.rand(16, 64, p)

    #     mps._initialize_cache(X)

    #     # sweep right to left
    #     n = mps.N-2
    #     while True:
    #         B = mps._merge(n)

    #         G1 = mps._gradient(n, B, X)
    #         G2 = mps._gradient(n, B, X, use_cache=True)

    #         self.assertTrue(np.allclose(G1, G2),
    #                         msg="gradient calculations with or without caching must agree at {} (right to left)".format(n))

    #         if n-1 > 1:
    #             self.assertTrue(self.is_left_canonical(mps, n-1),
    #                             msg="chain must be left canonical at {} before moving pivot".format(n-1))

    #             self.assertTrue(np.allclose(mps._contract_left(n-1, X), mps.cache[n-1]),
    #                             msg="cache must be consistent with left contraction at {} before moving pivot".format(n-1))

    #         if n+2 < mps.N:
    #             self.assertTrue(self.is_right_canonical(mps, n+2),
    #                             msg="chain must be right canonical at {} before moving pivot".format(n+2))

    #             self.assertTrue(np.allclose(mps._contract_right(n+2, X), mps.cache[n+2]),
    #                             msg="cache must be consistent with right contraction at {} before moving pivot".format(n-1))

    #         m = mps._move_pivot(X, n, 0.1, "right2left")

    #         self.assertTrue(m == n-1,
    #                         msg="pivot must move from right to left")

    #         if n-1 > 1:
    #             self.assertTrue(self.is_left_canonical(mps, n-1),
    #                             msg="chain must be left canonical at {} after moving pivot".format(n-1))

    #             self.assertTrue(np.allclose(mps._contract_left(n-1, X), mps.cache[n-1]),
    #                             msg="cache must be consistent with left contraction at {} after moving pivot".format(n-1))

    #         if n+2 < mps.N:
    #             self.assertTrue(self.is_right_canonical(mps, n+2),
    #                             msg="chain must be right canonical at {} after moving pivot".format(n+2))

    #             self.assertTrue(np.allclose(mps._contract_right(n+2, X), mps.cache[n+2]),
    #                             msg="cache must be consistent with right contraction at {} after moving pivot".format(n-1))

    #         n = m

    #         if n == -1:
    #             break


    #     # sweep right to left
    #     n = 0
    #     while True:
    #         B = mps._merge(n)

    #         G1 = mps._gradient(n, B, X)
    #         G2 = mps._gradient(n, B, X, use_cache=True)

    #         self.assertTrue(np.allclose(G1, G2),
    #                         msg="gradient calculations with or without caching must agree at {} (right to left)".format(n))

    #         if n-1 > 1:
    #             self.assertTrue(self.is_left_canonical(mps, n-1),
    #                             msg="chain must be left canonical at {} before moving pivot".format(n-1))

    #             self.assertTrue(np.allclose(mps._contract_left(n-1, X), mps.cache[n-1]),
    #                             msg="cache must be consistent with left contraction at {} before moving pivot".format(n-1))

    #         if n+2 < mps.N:
    #             self.assertTrue(self.is_right_canonical(mps, n+2),
    #                             msg="chain must be right canonical at {} before moving pivot".format(n+2))

    #             self.assertTrue(np.allclose(mps._contract_right(n+2, X), mps.cache[n+2]),
    #                             msg="cache must be consistent with right contraction at {} before moving pivot".format(n-1))

    #         m = mps._move_pivot(X, n, 0.1, "left2right")

    #         self.assertTrue(m == n+1,
    #                         msg="pivot must move from left to right")

    #         if n-1 > 1:
    #             self.assertTrue(self.is_left_canonical(mps, n-1),
    #                             msg="chain must be left canonical at {} after moving pivot".format(n-1))

    #             self.assertTrue(np.allclose(mps._contract_left(n-1, X), mps.cache[n-1]),
    #                             msg="cache must be consistent with left contraction at {} after moving pivot".format(n-1))

    #         if n+2 < mps.N:
    #             self.assertTrue(self.is_right_canonical(mps, n+2),
    #                             msg="chain must be right canonical at {} after moving pivot".format(n+2))

    #             self.assertTrue(np.allclose(mps._contract_right(n+2, X), mps.cache[n+2]),
    #                             msg="cache must be consistent with right contraction at {} after moving pivot".format(n-1))

    #         n = m

    #         if n == mps.N-1:
    #             break


    # def test_sweep_convergence(self):
    #     p = 5
    #     mps = Mps(N=64, part_d=p)

    #     self.assertTrue(self.is_left_canonical(mps, mps.N-2),
    #                     msg="before right to left sweep mps must be left canonical")

    #     X = np.random.rand(16, 64, p)

    #     mps._initialize_cache(X)

    #     l0 = mps.log_likelihood(X)

    #     # right to left pass
    #     n = mps.N-2
    #     while True:
    #         n = mps._move_pivot(X, n, 0.1, "right2left")
    #         if n == -1:
    #             break

    #     self.assertTrue(self.is_right_canonical(mps, 1),
    #                     msg="before left to right sweep mps must be right canonical")

    #     l1 = mps.log_likelihood(X)

    #     self.assertTrue(l1[1] < l0[1],
    #                     msg="log-likelihood must decrease after a right to left sweep")

    #     # left to right pass
    #     n = 0
    #     while True:
    #         n = mps._move_pivot(X, n, 0.1, "left2right")
    #         if n == mps.N-1:
    #             break

    #     self.assertTrue(self.is_left_canonical(mps, mps.N-2),
    #                     msg="after two sweeps mps must be back to left canonical")

    #     l2 = mps.log_likelihood(X)

    #     self.assertTrue(l2[1] < l1[1],
    #                     msg="log-likelihood must decrease after a left to right sweep")


if __name__ == '__main__':
    unittest.main()
