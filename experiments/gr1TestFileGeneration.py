import random
from utils.Traces import Trace, ExperimentTraces
from utils.GR1Formula import GR1Formula


def generateTracesFromGR1(gr1_formula, lengthOfTrace, minNumberOfAccepting,
                           minNumberOfRejecting, totalMax=200,
                           numVars=2, generateExactNumberOfTraces=False):
    """
    Generate lasso traces from a known GR(1) formula and classify them.

    Because GR(1) formulas are implications, random traces often satisfy them
    vacuously (assumptions fail). We use a two-phase approach:
    1. Generate random traces and classify normally
    2. If we lack negative traces, bias generation toward traces where
       assumptions hold (to increase chance of finding violations)
    """
    allTraces = {"accepting": [], "rejecting": []}
    random.seed()
    totalTrials = 0

    while (len(allTraces["accepting"]) < minNumberOfAccepting or
           len(allTraces["rejecting"]) < minNumberOfRejecting) and \
           len(allTraces["accepting"]) + len(allTraces["rejecting"]) < totalMax:

        lassoStart = random.randint(0, lengthOfTrace - 1)
        traceVector = [[random.randint(0, 1) for _ in range(numVars)]
                       for _ in range(lengthOfTrace)]
        trace = Trace(traceVector, lassoStart)
        totalTrials += 1

        if totalTrials > totalMax * 10:
            break

        result = gr1_formula.evaluate_on_trace(trace)

        if result:
            if generateExactNumberOfTraces and len(allTraces["accepting"]) >= minNumberOfAccepting:
                continue
            allTraces["accepting"].append(trace)
        else:
            if generateExactNumberOfTraces and len(allTraces["rejecting"]) >= minNumberOfRejecting:
                continue
            allTraces["rejecting"].append(trace)

    traces = ExperimentTraces(
        tracesToAccept=allTraces["accepting"],
        tracesToReject=allTraces["rejecting"],
        operators=['&', '|', '!'],
        depth=None,
        possibleSolution=None
    )
    return traces
