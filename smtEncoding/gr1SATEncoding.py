from z3 import *
from utils.SimpleTree import Formula
from utils.GR1Formula import GR1Formula


class GR1SATEncoding:
    """
    SAT encoding for mining GR(1) specifications of the form:
        (Init_e ∧ □Safety_e ∧ □◇J_1 ∧ ... ∧ □◇J_m) → (Init_s ∧ □Safety_s ∧ □◇G_1 ∧ ... ∧ □◇G_k)

    Each component is a propositional formula encoded as a literal-leaves tree:
    D leaf nodes (each a variable with a polarity flag for negation) connected
    by D-1 internal binary operator nodes (& or |).  The temporal skeleton is
    fixed — only propositional structure is searched.

    D=1: single literal (x or ¬x).
    D=2: two literals combined by one binary op.
    D=k: k literals, k-1 binary operators.

    Implements the same interface as DagSATEncoding so it can be used with
    the existing satQuerying.py solving loop.
    """

    def __init__(self, D, testTraces, num_justices=1, num_guarantees=1,
                 include_init=False, include_safety=False, env_var_indices=None):
        self.formulaDepth = D
        self.traces = testTraces
        self.num_justices = num_justices
        self.num_guarantees = num_guarantees
        self.include_init = include_init
        self.include_safety = include_safety

        # Env/sys variable partition for realizability
        n = self.traces.numVariables
        if env_var_indices is not None:
            self.env_var_indices = list(env_var_indices)
            self.sys_var_indices = [i for i in range(n) if i not in set(env_var_indices)]
        else:
            self.env_var_indices = None
            self.sys_var_indices = None

        self.solver = Solver()

        self.listOfVariables = list(range(n))
        self.binaryOperators = ['&', '|']

        # Component names
        self.justice_names = ['J_%d' % i for i in range(num_justices)]
        self.guarantee_names = ['G_%d' % i for i in range(num_guarantees)]

        self.init_names = []
        self.safety_names = []
        if include_init:
            self.init_names = ['init_e', 'init_s']
        if include_safety:
            self.safety_names = ['safety_e', 'safety_s']

        self.assumption_components = (
            [n for n in self.init_names if n.endswith('_e')] +
            [n for n in self.safety_names if n.endswith('_e')] +
            self.justice_names
        )
        self.guarantee_components = (
            [n for n in self.init_names if n.endswith('_s')] +
            [n for n in self.safety_names if n.endswith('_s')] +
            self.guarantee_names
        )

        self.component_names = (self.init_names + self.safety_names +
                                self.justice_names + self.guarantee_names)

        # Per-component Z3 variable dictionaries
        self.x = {}
        self.l = {}
        self.r = {}
        self.y = {}
        self.neg = {}
        self.holds = {}

        self.allTraces = self.traces.acceptedTraces + self.traces.rejectedTraces

    def _get_variables_for_component(self, comp):
        """Return leaf variable indices for a component.

        Without a partition (env_var_indices=None): original behaviour —
        safety gets all current + all primed vars, others get all current.

        With a partition, enforce GR(1) realizability constraints:
          - safety_e primed vars: env only   | current: all
          - safety_s primed vars: sys only   | current: all
          - justice  (J_i):  env current vars only
          - guarantee (G_i): sys current vars only
          - init_e: env current vars only
          - init_s: sys current vars only
        """
        n = self.traces.numVariables

        if self.env_var_indices is None:
            # No partition — original behavior
            if comp in self.safety_names:
                return list(range(2 * n))
            return list(range(n))

        # Partition-aware variable selection
        if comp == 'safety_e':
            # Current: all vars | Primed: env only
            return list(range(n)) + [n + i for i in self.env_var_indices]
        elif comp == 'safety_s':
            # Current: all vars | Primed: sys only
            return list(range(n)) + [n + i for i in self.sys_var_indices]
        elif comp in self.justice_names:
            return list(self.env_var_indices)
        elif comp in self.guarantee_names:
            return list(self.sys_var_indices)
        elif comp == 'init_e':
            return list(self.env_var_indices)
        elif comp == 'init_s':
            return list(self.sys_var_indices)
        else:
            return list(range(n))

    def _total_nodes(self):
        """Total nodes in the literal-leaves tree: max(1, 2*D - 1)."""
        return max(1, 2 * self.formulaDepth - 1)

    def _root_index(self):
        """Index of the root node."""
        if self.formulaDepth <= 1:
            return 0
        return 2 * self.formulaDepth - 2

    def getInformativeVariables(self):
        res = []
        for comp in self.component_names:
            res += list(self.x[comp].values())
            res += list(self.l[comp].values())
            res += list(self.r[comp].values())
            res += list(self.neg[comp].values())
        return res

    def encodeFormula(self, unsatCore=True):
        self.solver.set(unsat_core=unsatCore)

        self.encodeAllComponents(unsatCore=False)
        self._addTemporalHoldsConstraints()
        self._addConsistencyConstraints(
            self.assumption_components, self.guarantee_components)

    def encodeAllComponents(self, unsatCore=True):
        """Encode structural + semantic constraints for all components.

        Does NOT add temporal holds or consistency constraints — call
        _addTemporalHoldsConstraints() and _addConsistencyConstraints()
        separately. This split enables incremental solving: encode once,
        then push/pop different consistency constraints per template config.
        """
        if unsatCore:
            self.solver.set(unsat_core=True)

        for comp in self.component_names:
            self._createComponentVariables(comp)

        for comp in self.component_names:
            self._addStructuralConstraints(comp)

        for comp in self.component_names:
            self._addPropSemantics(comp)

    def _createComponentVariables(self, comp):
        D = self.formulaDepth
        total = self._total_nodes()
        comp_vars = self._get_variables_for_component(comp)

        # x: leaves select a variable, internals select a binary operator
        x_dict = {}
        for i in range(D):
            for v in comp_vars:
                x_dict[(i, v)] = Bool('x_%s_%d_%s' % (comp, i, str(v)))
        for i in range(D, total):
            for op in self.binaryOperators:
                x_dict[(i, op)] = Bool('x_%s_%d_%s' % (comp, i, op))
        self.x[comp] = x_dict

        # neg: polarity flag for each leaf (True = negated literal)
        self.neg[comp] = {
            i: Bool('neg_%s_%d' % (comp, i))
            for i in range(D)
        }

        # l, r: only for internal nodes
        self.l[comp] = {
            (i, j): Bool('l_%s_%d_%d' % (comp, i, j))
            for i in range(D, total) for j in range(i)
        }
        self.r[comp] = {
            (i, j): Bool('r_%s_%d_%d' % (comp, i, j))
            for i in range(D, total) for j in range(i)
        }

        # y: all nodes (leaves + internals)
        self.y[comp] = {
            (i, trIdx, t): Bool('y_%s_%d_%d_%d' % (comp, i, trIdx, t))
            for i in range(total)
            for trIdx, tr in enumerate(self.allTraces)
            for t in range(tr.lengthOfTrace)
        }
        self.holds[comp] = {
            trIdx: Bool('holds_%s_%d' % (comp, trIdx))
            for trIdx in range(len(self.allTraces))
        }

    def _addStructuralConstraints(self, comp):
        """Add structural constraints for the literal-leaves encoding.

        Leaf nodes (0..D-1): exactly one variable selected (polarity is free).
        Internal nodes (D..2D-2): exactly one binary operator, exactly one left/right child.
        No-dangling: every non-root node must be referenced as a child.
        """
        D = self.formulaDepth
        total = self._total_nodes()
        x = self.x[comp]
        l = self.l[comp]
        r = self.r[comp]
        comp_vars = self._get_variables_for_component(comp)

        # Leaf nodes: exactly one variable
        for i in range(D):
            node_vars = [x[(i, v)] for v in comp_vars]
            self.solver.assert_and_track(
                AtMost(*node_vars, 1),
                'atmost_one_var_%s_%d' % (comp, i)
            )
            self.solver.assert_and_track(
                AtLeast(*node_vars, 1),
                'atleast_one_var_%s_%d' % (comp, i)
            )

        # Internal nodes: exactly one operator, exactly one left/right child
        for i in range(D, total):
            op_vars = [x[(i, op)] for op in self.binaryOperators]
            self.solver.assert_and_track(
                AtMost(*op_vars, 1),
                'atmost_one_op_%s_%d' % (comp, i)
            )
            self.solver.assert_and_track(
                AtLeast(*op_vars, 1),
                'atleast_one_op_%s_%d' % (comp, i)
            )

            left_vars = [l[(i, j)] for j in range(i)]
            self.solver.assert_and_track(
                AtMost(*left_vars, 1),
                'atmost_one_left_%s_%d' % (comp, i)
            )
            self.solver.assert_and_track(
                AtLeast(*left_vars, 1),
                'atleast_one_left_%s_%d' % (comp, i)
            )

            right_vars = [r[(i, j)] for j in range(i)]
            self.solver.assert_and_track(
                AtMost(*right_vars, 1),
                'atmost_one_right_%s_%d' % (comp, i)
            )
            self.solver.assert_and_track(
                AtLeast(*right_vars, 1),
                'atleast_one_right_%s_%d' % (comp, i)
            )

        # No dangling nodes: every non-root node must be someone's child
        if total > 1:
            root = total - 1
            for i in range(root):
                parents_l = [l[(rid, i)] for rid in range(D, total) if (rid, i) in l]
                parents_r = [r[(rid, i)] for rid in range(D, total) if (rid, i) in r]
                parent_refs = parents_l + parents_r
                if parent_refs:
                    self.solver.assert_and_track(
                        Or(parent_refs),
                        'no_dangling_%s_%d' % (comp, i)
                    )

    def _addPropSemantics(self, comp):
        """Add propositional semantics for literal-leaves encoding.

        Leaves: y = trace_value XOR neg (polarity flag).
        Internal nodes: y = left OP right (only & and |).
        """
        D = self.formulaDepth
        total = self._total_nodes()
        x = self.x[comp]
        l = self.l[comp]
        r = self.r[comp]
        y = self.y[comp]
        neg = self.neg[comp]
        comp_vars = self._get_variables_for_component(comp)
        n = self.traces.numVariables

        for trIdx, tr in enumerate(self.allTraces):
            # Leaf semantics (nodes 0..D-1) with polarity
            for i in range(D):
                for p in comp_vars:
                    if p < n:
                        self.solver.assert_and_track(
                            Implies(
                                x[(i, p)],
                                And([
                                    y[(i, trIdx, t)] == (
                                        Not(neg[i]) if tr.traceVector[t][p]
                                        else neg[i]
                                    )
                                    for t in range(tr.lengthOfTrace)
                                ])
                            ),
                            'var_sem_%s_%d_%d_%d' % (comp, i, p, trIdx)
                        )
                    else:
                        orig_var = p - n
                        self.solver.assert_and_track(
                            Implies(
                                x[(i, p)],
                                And([
                                    y[(i, trIdx, t)] == (
                                        Not(neg[i]) if tr.traceVector[tr.nextPos(t)][orig_var]
                                        else neg[i]
                                    )
                                    for t in range(tr.lengthOfTrace)
                                ])
                            ),
                            'var_sem_prime_%s_%d_%d_%d' % (comp, i, p, trIdx)
                        )

            # Internal semantics (nodes D..total-1): only & and |
            for i in range(D, total):
                self.solver.assert_and_track(
                    Implies(
                        x[(i, '&')],
                        And([
                            Implies(
                                And(l[(i, la)], r[(i, ra)]),
                                And([
                                    y[(i, trIdx, t)] ==
                                    And(y[(la, trIdx, t)], y[(ra, trIdx, t)])
                                    for t in range(tr.lengthOfTrace)
                                ])
                            )
                            for la in range(i) for ra in range(i)
                        ])
                    ),
                    'and_sem_%s_%d_%d' % (comp, i, trIdx)
                )

                self.solver.assert_and_track(
                    Implies(
                        x[(i, '|')],
                        And([
                            Implies(
                                And(l[(i, la)], r[(i, ra)]),
                                And([
                                    y[(i, trIdx, t)] ==
                                    Or(y[(la, trIdx, t)], y[(ra, trIdx, t)])
                                    for t in range(tr.lengthOfTrace)
                                ])
                            )
                            for la in range(i) for ra in range(i)
                        ])
                    ),
                    'or_sem_%s_%d_%d' % (comp, i, trIdx)
                )

    def _addTemporalHoldsConstraints(self):
        """Add temporal semantics: connect holds[comp] to y[comp] at root.

        These are per-component and independent of which template config is
        active, so they are encoded once and persist across push/pop.
        """
        root = self._root_index()

        for trIdx, tr in enumerate(self.allTraces):
            loop_positions = list(range(tr.lassoStart, tr.lengthOfTrace))
            all_positions = list(range(tr.lengthOfTrace))

            for name in self.init_names:
                self.solver.assert_and_track(
                    self.holds[name][trIdx] ==
                    self.y[name][(root, trIdx, 0)],
                    'init_holds_%s_%d' % (name, trIdx)
                )

            for name in self.safety_names:
                self.solver.assert_and_track(
                    self.holds[name][trIdx] ==
                    And([self.y[name][(root, trIdx, t)] for t in all_positions]),
                    'safety_holds_%s_%d' % (name, trIdx)
                )

            for jname in self.justice_names:
                self.solver.assert_and_track(
                    self.holds[jname][trIdx] ==
                    Or([self.y[jname][(root, trIdx, t)] for t in loop_positions]),
                    'justice_holds_%s_%d' % (jname, trIdx)
                )

            for gname in self.guarantee_names:
                self.solver.assert_and_track(
                    self.holds[gname][trIdx] ==
                    Or([self.y[gname][(root, trIdx, t)] for t in loop_positions]),
                    'guarantee_holds_%s_%d' % (gname, trIdx)
                )

    def _addConsistencyConstraints(self, assumption_comps, guarantee_comps):
        """Add trace classification constraints for a specific template config.

        In incremental mode, these are added inside a push/pop scope so they
        can be swapped out for the next template configuration.
        """
        numAccepted = len(self.traces.acceptedTraces)

        for trIdx in range(numAccepted):
            assumptions = And([self.holds[c][trIdx] for c in assumption_comps])
            guarantees = And([self.holds[c][trIdx] for c in guarantee_comps])
            self.solver.assert_and_track(
                Implies(assumptions, guarantees),
                'positive_trace_%d' % trIdx
            )

        for trIdx in range(numAccepted, len(self.allTraces)):
            assumptions = And([self.holds[c][trIdx] for c in assumption_comps])
            guarantees = And([self.holds[c][trIdx] for c in guarantee_comps])
            self.solver.assert_and_track(
                And(assumptions, Not(guarantees)),
                'negative_trace_%d' % trIdx
            )

    def _addSymmetryBreaking(self):
        """Add symmetry-breaking constraints to prune equivalent solutions."""
        for comp in self.component_names:
            self._addCommutativeChildOrdering(comp)
        self._addComponentPermutationBreaking(self.justice_names)
        self._addComponentPermutationBreaking(self.guarantee_names)

    def _addCommutativeChildOrdering(self, comp):
        """For commutative operators (& and |), enforce left_child ≤ right_child."""
        D = self.formulaDepth
        total = self._total_nodes()
        if total <= 1:
            return
        x = self.x[comp]
        l = self.l[comp]
        r = self.r[comp]

        for i in range(D, total):
            for la in range(i):
                for ra in range(i):
                    if ra < la:
                        self.solver.add(
                            Implies(
                                Or(x[(i, '&')], x[(i, '|')]),
                                Not(And(l[(i, la)], r[(i, ra)]))
                            )
                        )

    def _addComponentPermutationBreaking(self, component_names):
        """Order interchangeable components lexicographically by structure.

        Justice components J_0,...,J_m are conjoined (□◇J_0 ∧ ... ∧ □◇J_m),
        so any permutation gives an equivalent formula. Same for guarantees.

        We break this by requiring that the operator/variable chosen at the root
        of component i is ≤ that of component i+1.
        """
        if len(component_names) < 2:
            return
        D = self.formulaDepth
        root = self._root_index()

        for idx in range(len(component_names) - 1):
            c1 = component_names[idx]
            c2 = component_names[idx + 1]

            if D <= 1:
                # Root is a leaf: order by variable index
                comp_vars = self._get_variables_for_component(c1)
                for pos1, v1 in enumerate(comp_vars):
                    for pos2, v2 in enumerate(comp_vars):
                        if pos1 > pos2:
                            self.solver.add(
                                Not(And(
                                    self.x[c1][(root, v1)],
                                    self.x[c2][(root, v2)]
                                ))
                            )
            else:
                # Root is internal: order by operator
                for pos1, o1 in enumerate(self.binaryOperators):
                    for pos2, o2 in enumerate(self.binaryOperators):
                        if pos1 > pos2:
                            self.solver.add(
                                Not(And(
                                    self.x[c1][(root, o1)],
                                    self.x[c2][(root, o2)]
                                ))
                            )

    def reconstructForConfig(self, model, active_justices, active_guarantees,
                             has_init, has_safety):
        """Reconstruct a GR1Formula from a model for a specific template config.

        Used by the incremental solver where the encoder has all possible
        components but only a subset is active in the current config.
        """
        root = self._root_index()
        justices = [self._reconstructPropFormula(j, root, model)
                    for j in active_justices]
        guarantees = [self._reconstructPropFormula(g, root, model)
                      for g in active_guarantees]

        init_e = (self._reconstructPropFormula('init_e', root, model)
                  if has_init else None)
        init_s = (self._reconstructPropFormula('init_s', root, model)
                  if has_init else None)
        safety_e = (self._reconstructPropFormula('safety_e', root, model)
                    if has_safety else None)
        safety_s = (self._reconstructPropFormula('safety_s', root, model)
                    if has_safety else None)

        return GR1Formula(
            justices=justices, guarantees=guarantees,
            init_e=init_e, init_s=init_s,
            safety_e=safety_e, safety_s=safety_s
        )

    def reconstructWholeFormula(self, model):
        justices = [self._reconstructPropFormula(j, self._root_index(), model)
                    for j in self.justice_names]
        guarantees = [self._reconstructPropFormula(g, self._root_index(), model)
                      for g in self.guarantee_names]

        init_e = (self._reconstructPropFormula('init_e', self._root_index(), model)
                  if self.include_init else None)
        init_s = (self._reconstructPropFormula('init_s', self._root_index(), model)
                  if self.include_init else None)
        safety_e = (self._reconstructPropFormula('safety_e', self._root_index(), model)
                    if self.include_safety else None)
        safety_s = (self._reconstructPropFormula('safety_s', self._root_index(), model)
                    if self.include_safety else None)

        return GR1Formula(
            justices=justices, guarantees=guarantees,
            init_e=init_e, init_s=init_s,
            safety_e=safety_e, safety_s=safety_s
        )

    def _reconstructPropFormula(self, comp, rowId, model):
        """Reconstruct the propositional formula for a component from the Z3 model."""
        def getValue(row, var_dict):
            tt = [k[1] for k in var_dict if k[0] == row and model[var_dict[k]] == True]
            if len(tt) > 1:
                raise Exception("more than one true value for %s row %d" % (comp, row))
            if len(tt) == 0:
                raise Exception("no true value for %s row %d" % (comp, row))
            return tt[0]

        D = self.formulaDepth
        x = self.x[comp]
        l = self.l[comp]
        r = self.r[comp]
        n = self.traces.numVariables

        if rowId < D:
            # Leaf node: get variable and polarity
            var_idx = getValue(rowId, x)
            is_negated = (model[self.neg[comp][rowId]] == True)
            if var_idx < n:
                base = Formula('x' + str(var_idx))
            else:
                base = Formula('x' + str(var_idx - n) + "'")
            if is_negated:
                return Formula(['!', base])
            return base
        else:
            # Internal node: binary operator with two children
            operator = getValue(rowId, x)
            leftChild = getValue(rowId, l)
            rightChild = getValue(rowId, r)
            return Formula([operator,
                            self._reconstructPropFormula(comp, leftChild, model),
                            self._reconstructPropFormula(comp, rightChild, model)])
