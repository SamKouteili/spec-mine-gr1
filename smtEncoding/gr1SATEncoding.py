from z3 import *
from utils.SimpleTree import Formula
from utils.GR1Formula import GR1Formula


class GR1SATEncoding:
    """
    SAT encoding for mining GR(1) specifications of the form:
        (Init_e ∧ □Safety_e ∧ □◇J_1 ∧ ... ∧ □◇J_m) → (Init_s ∧ □Safety_s ∧ □◇G_1 ∧ ... ∧ □◇G_k)

    Each component is a propositional formula encoded as a mini-DAG of size D.
    The temporal skeleton is fixed — only propositional operators (&, |, !) are searched.

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
        self.propOperators = ['&', '|', '!']
        self.unaryOperators = ['!']
        self.binaryOperators = ['&', '|']
        self.operatorsAndVariables = self.propOperators + self.listOfVariables

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

    def _get_ops_and_vars(self, comp):
        """Return operators + variables list for a component."""
        return self.propOperators + self._get_variables_for_component(comp)

    def getInformativeVariables(self):
        res = []
        for comp in self.component_names:
            res += list(self.x[comp].values())
            res += list(self.l[comp].values())
            res += list(self.r[comp].values())
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
        comp_ops_and_vars = self._get_ops_and_vars(comp)

        self.x[comp] = {
            (i, o): Bool('x_%s_%d_%s' % (comp, i, str(o)))
            for i in range(D) for o in comp_ops_and_vars
        }
        self.l[comp] = {
            (i, j): Bool('l_%s_%d_%d' % (comp, i, j))
            for i in range(1, D) for j in range(i)
        }
        self.r[comp] = {
            (i, j): Bool('r_%s_%d_%d' % (comp, i, j))
            for i in range(1, D) for j in range(i)
        }
        self.y[comp] = {
            (i, trIdx, t): Bool('y_%s_%d_%d_%d' % (comp, i, trIdx, t))
            for i in range(D)
            for trIdx, tr in enumerate(self.allTraces)
            for t in range(tr.lengthOfTrace)
        }
        self.holds[comp] = {
            trIdx: Bool('holds_%s_%d' % (comp, trIdx))
            for trIdx in range(len(self.allTraces))
        }

    def _addStructuralConstraints(self, comp):
        """Add DAG structural constraints for a component (propositional operators only)."""
        D = self.formulaDepth
        x = self.x[comp]
        l = self.l[comp]
        r = self.r[comp]
        comp_vars = self._get_variables_for_component(comp)
        comp_ops_and_vars = self._get_ops_and_vars(comp)

        # Exactly one operator/variable per node
        for i in range(D):
            node_vars = [x[(i, o)] for o in comp_ops_and_vars]
            self.solver.assert_and_track(
                AtMost(*node_vars, 1),
                'atmost_one_op_%s_%d' % (comp, i)
            )
            self.solver.assert_and_track(
                AtLeast(*node_vars, 1),
                'atleast_one_op_%s_%d' % (comp, i)
            )

        # Node 0 must be a propositional variable
        self.solver.assert_and_track(
            Or([x[(0, v)] for v in comp_vars]),
            'first_is_var_%s' % comp
        )

        if D <= 1:
            return

        # No dangling nodes
        self.solver.assert_and_track(
            And([
                Or(
                    Or([l[(rowId, i)] for rowId in range(i + 1, D)]),
                    Or([r[(rowId, i)] for rowId in range(i + 1, D)])
                )
                for i in range(D - 1)
            ]),
            'no_dangling_%s' % comp
        )

        for i in range(1, D):
            # Binary operators (&, |): exactly one left and right child
            self.solver.assert_and_track(
                Implies(
                    Or([x[(i, op)] for op in self.binaryOperators]),
                    And(
                        AtMost(*[l[(i, j)] for j in range(i)], 1),
                        AtLeast(*[l[(i, j)] for j in range(i)], 1),
                        AtMost(*[r[(i, j)] for j in range(i)], 1),
                        AtLeast(*[r[(i, j)] for j in range(i)], 1),
                    )
                ),
                'binary_children_%s_%d' % (comp, i)
            )

            # Unary operator (!): exactly one left child, no right child
            self.solver.assert_and_track(
                Implies(
                    Or([x[(i, op)] for op in self.unaryOperators]),
                    And(
                        AtMost(*[l[(i, j)] for j in range(i)], 1),
                        AtLeast(*[l[(i, j)] for j in range(i)], 1),
                        Not(Or([r[(i, j)] for j in range(i)])),
                    )
                ),
                'unary_children_%s_%d' % (comp, i)
            )

            # Variables: no children
            self.solver.assert_and_track(
                Implies(
                    Or([x[(i, v)] for v in comp_vars]),
                    And(
                        Not(Or([l[(i, j)] for j in range(i)])),
                        Not(Or([r[(i, j)] for j in range(i)])),
                    )
                ),
                'var_no_children_%s_%d' % (comp, i)
            )

    def _addPropSemantics(self, comp):
        """Add propositional semantics constraints for a component."""
        D = self.formulaDepth
        x = self.x[comp]
        l = self.l[comp]
        r = self.r[comp]
        y = self.y[comp]
        comp_vars = self._get_variables_for_component(comp)
        n = self.traces.numVariables

        for trIdx, tr in enumerate(self.allTraces):
            for i in range(D):
                for p in comp_vars:
                    if p < n:
                        # Current-state variable: value at time t
                        self.solver.assert_and_track(
                            Implies(
                                x[(i, p)],
                                And([
                                    y[(i, trIdx, t)] if tr.traceVector[t][p]
                                    else Not(y[(i, trIdx, t)])
                                    for t in range(tr.lengthOfTrace)
                                ])
                            ),
                            'var_sem_%s_%d_%d_%d' % (comp, i, p, trIdx)
                        )
                    else:
                        # Primed variable: value from successor position
                        orig_var = p - n
                        self.solver.assert_and_track(
                            Implies(
                                x[(i, p)],
                                And([
                                    y[(i, trIdx, t)] if tr.traceVector[tr.nextPos(t)][orig_var]
                                    else Not(y[(i, trIdx, t)])
                                    for t in range(tr.lengthOfTrace)
                                ])
                            ),
                            'var_sem_prime_%s_%d_%d_%d' % (comp, i, p, trIdx)
                        )

            for i in range(1, D):
                # Conjunction
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

                # Disjunction
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

                # Negation
                self.solver.assert_and_track(
                    Implies(
                        x[(i, '!')],
                        And([
                            Implies(
                                l[(i, child)],
                                And([
                                    y[(i, trIdx, t)] == Not(y[(child, trIdx, t)])
                                    for t in range(tr.lengthOfTrace)
                                ])
                            )
                            for child in range(i)
                        ])
                    ),
                    'not_sem_%s_%d_%d' % (comp, i, trIdx)
                )

    def _addTemporalHoldsConstraints(self):
        """Add temporal semantics: connect holds[comp] to y[comp] at root.

        These are per-component and independent of which template config is
        active, so they are encoded once and persist across push/pop.
        """
        D = self.formulaDepth

        for trIdx, tr in enumerate(self.allTraces):
            loop_positions = list(range(tr.lassoStart, tr.lengthOfTrace))
            all_positions = list(range(tr.lengthOfTrace))

            for name in self.init_names:
                self.solver.assert_and_track(
                    self.holds[name][trIdx] ==
                    self.y[name][(D - 1, trIdx, 0)],
                    'init_holds_%s_%d' % (name, trIdx)
                )

            for name in self.safety_names:
                self.solver.assert_and_track(
                    self.holds[name][trIdx] ==
                    And([self.y[name][(D - 1, trIdx, t)] for t in all_positions]),
                    'safety_holds_%s_%d' % (name, trIdx)
                )

            for jname in self.justice_names:
                self.solver.assert_and_track(
                    self.holds[jname][trIdx] ==
                    Or([self.y[jname][(D - 1, trIdx, t)] for t in loop_positions]),
                    'justice_holds_%s_%d' % (jname, trIdx)
                )

            for gname in self.guarantee_names:
                self.solver.assert_and_track(
                    self.holds[gname][trIdx] ==
                    Or([self.y[gname][(D - 1, trIdx, t)] for t in loop_positions]),
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
            self._addDoubleNegationElimination(comp)
        self._addComponentPermutationBreaking(self.justice_names)
        self._addComponentPermutationBreaking(self.guarantee_names)

    def _addCommutativeChildOrdering(self, comp):
        """For commutative operators (& and |), enforce left_child ≤ right_child.

        The formulas (a & b) and (b & a) are equivalent. By requiring the left
        child index to be ≤ the right child index, we eliminate half the search
        space for every binary node.
        """
        D = self.formulaDepth
        if D <= 1:
            return
        x = self.x[comp]
        l = self.l[comp]
        r = self.r[comp]

        for i in range(1, D):
            for la in range(i):
                for ra in range(i):
                    if ra < la:
                        # If node i is commutative and has left=la, right=ra with ra < la,
                        # that's a symmetry violation — block it
                        self.solver.add(
                            Implies(
                                Or(x[(i, '&')], x[(i, '|')]),
                                Not(And(l[(i, la)], r[(i, ra)]))
                            )
                        )

    def _addDoubleNegationElimination(self, comp):
        """Prevent ¬(¬φ) — if node i is !, its child cannot be !.

        Double negation is always redundant, so any solution containing it
        has an equivalent smaller solution without it.
        """
        D = self.formulaDepth
        if D <= 2:
            return
        x = self.x[comp]
        l = self.l[comp]

        for i in range(1, D):
            for child in range(i):
                if child >= 1:  # child must be an internal node to be !
                    self.solver.add(
                        Implies(
                            And(x[(i, '!')], l[(i, child)]),
                            Not(x[(child, '!')])
                        )
                    )

    def _addComponentPermutationBreaking(self, component_names):
        """Order interchangeable components lexicographically by structure.

        Justice components J_0,...,J_m are conjoined (□◇J_0 ∧ ... ∧ □◇J_m),
        so any permutation gives an equivalent formula. Same for guarantees.

        We break this by requiring that the operator/variable chosen at the root
        of component i is ≤ that of component i+1 (using the index in the
        operators-and-variables list as the ordering).
        """
        if len(component_names) < 2:
            return
        D = self.formulaDepth
        root = D - 1

        for idx in range(len(component_names) - 1):
            c1 = component_names[idx]
            c2 = component_names[idx + 1]
            ops_vars = self._get_ops_and_vars(c1)

            # For each pair (o1, o2) where o1 appears after o2 in the ordering,
            # block c1=o1 and c2=o2 at the root
            for pos1, o1 in enumerate(ops_vars):
                for pos2, o2 in enumerate(ops_vars):
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
        root = self.formulaDepth - 1
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
        justices = [self._reconstructPropFormula(j, self.formulaDepth - 1, model)
                    for j in self.justice_names]
        guarantees = [self._reconstructPropFormula(g, self.formulaDepth - 1, model)
                      for g in self.guarantee_names]

        init_e = (self._reconstructPropFormula('init_e', self.formulaDepth - 1, model)
                  if self.include_init else None)
        init_s = (self._reconstructPropFormula('init_s', self.formulaDepth - 1, model)
                  if self.include_init else None)
        safety_e = (self._reconstructPropFormula('safety_e', self.formulaDepth - 1, model)
                    if self.include_safety else None)
        safety_s = (self._reconstructPropFormula('safety_s', self.formulaDepth - 1, model)
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

        x = self.x[comp]
        l = self.l[comp]
        r = self.r[comp]
        comp_vars = self._get_variables_for_component(comp)
        n = self.traces.numVariables

        operator = getValue(rowId, x)

        if operator in comp_vars:
            if operator < n:
                return Formula('x' + str(operator))
            else:
                return Formula('x' + str(operator - n) + "'")
        elif operator in self.unaryOperators:
            leftChild = getValue(rowId, l)
            return Formula([operator, self._reconstructPropFormula(comp, leftChild, model)])
        elif operator in self.binaryOperators:
            leftChild = getValue(rowId, l)
            rightChild = getValue(rowId, r)
            return Formula([operator,
                            self._reconstructPropFormula(comp, leftChild, model),
                            self._reconstructPropFormula(comp, rightChild, model)])
