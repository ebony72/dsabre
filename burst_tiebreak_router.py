"""
dSABRE_BurstTiebreak — dSABRE with a boolean burst tiebreaker in candidate scoring.

Idea (Option C — "boolean burst tiebreaker"):
  When two teleport candidates have equal (or very close) scores, prefer the
  one that would immediately unlock a front-layer gate: i.e., after moving
  virt to lq_dst, some gate in front_inter has its partner already in
  next_core and physically adjacent to lq_dst.

  The tiebreak is purely binary (0 or 1) and costs nothing extra: no ECH,
  no new hyperparameter.  Ties are broken by:
    primary:   score   (lower = better)
    secondary: -burst  (burst=1 preferred over burst=0)
"""

from router import General_dSABRE_Router


class dSABRE_BurstTiebreak(General_dSABRE_Router):

    def _burst_now(self, virt, lq_dst, next_core, front_inter, l2p):
        arch = self.arch
        for node in front_inter:
            q1, q2 = node.qargs[0], node.qargs[1]
            if virt not in (q1, q2):
                continue
            partner = q2 if virt == q1 else q1
            p_partner = l2p[partner]
            if (arch.core_of(p_partner) == next_core
                    and arch.Gr.has_edge(lq_dst, p_partner)):
                return 1
        return 0

    def _generate_candidates(self, front_inter, extended, l2p, p2l):
        candidates = super()._generate_candidates(front_inter, extended, l2p, p2l)
        # Re-sort with burst tiebreaker; parent already sorted by score alone.
        candidates.sort(key=lambda a: (
            a.score,
            -self._burst_now(a.virt, a.p_comm_dst, a.next_core, front_inter, l2p),
        ))
        return candidates
