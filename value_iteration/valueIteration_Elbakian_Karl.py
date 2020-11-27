import numpy as np
import drawHeatMap as hm
import rewardTable as rt
import transitionTable as tt


def expect(xDistribution, function):
    expectation=sum([function(x)*px for x, px in xDistribution.items()])
    return expectation

def getSPrimeRDistributionFull(s, action, transitionTable, rewardTable): 
    reward=lambda sPrime: rewardTable[s][action][sPrime]
    p=lambda sPrime: transitionTable[s][action][sPrime]
    sPrimeRDistribution={(sPrime, reward(sPrime)): p(sPrime) for sPrime in transitionTable[s][action].keys()}
    return sPrimeRDistribution

#Function to use the bellman equation as an update rule. This functions is used to determine new V[s], and also help determine the policy.
def bellmanUpdate(s, V, transitionTable, rewardTable, getSPrimeRDistributionFull, gamma, roundingTolerance):
    Q = {}
    policy = {}
    max_eu_actions = list()
    A = list(transitionTable[s].keys())
    Q = {action:expect(getSPrimeRDistributionFull(s, action, transitionTable, rewardTable), lambda sPrime: sPrime[1] + gamma*V[sPrime[0]]) for action in A}
    updated_v =  max(Q.values()) #update rule, where new value of state is the maximum EU over all actions

    #Consider EU of actions within the rounding tolerance to be the same as the max EU
    max_eu_actions = [action for action in Q.keys() if abs(Q[action] - updated_v) < roundingTolerance]

    #Defining policy. Considered to be 1/#a, where #a is the number of actions that are considered to have max EU.
    policy = {action: 1/len(max_eu_actions) for action in max_eu_actions} 
    return policy, updated_v

def main():
    
    minX, maxX, minY, maxY=(0, 3, 0, 2)
    convergenceTolerance = 10e-7
    roundingTolerance= 10e-7
    gamma = 0.8
    
    possibleAction=[(0,1), (0,-1), (1,0), (-1,0)]
    possibleState=[(i,j) for i in range(maxX+1) for j in range(maxY+1)]
    V={s:0 for s in possibleState}
    
    normalCost=-0.04
    trapDict={(3,1):-1}
    bonusDict={(3,0):1}
    blockList=[(1,1)]
    
    p=0.8
    transitionProbability={'forward':p, 'left':(1-p)/2, 'right':(1-p)/2, 'back':0}
    transitionProbability={move: p for move, p in transitionProbability.items() if transitionProbability[move]!=0}
    
    transitionTable=tt.createTransitionTable(minX, minY, maxX, maxY, trapDict, bonusDict, blockList, possibleAction, transitionProbability)
    rewardTable=rt.createRewardTable(transitionTable, normalCost, trapDict, bonusDict)

    def valueIteration(V, transitionTable, rewardTable, convergenceTolerance, gamma, bellmanUpdate, getSPrimeRDistributionFull, roundingTolerance):
        policy = {}
        deltas={s:np.Inf for s in transitionTable.keys()}
        #iterate through states, updating V[s] until difference in delta is less than convergence tolerance.
        while max(deltas.values()) > convergenceTolerance:
            for s in transitionTable.keys():
                deltas[s] = 0
                v = V[s]
                _ , V[s] = bellmanUpdate(s, V, transitionTable, rewardTable, getSPrimeRDistributionFull, gamma, roundingTolerance)
                deltas[s] = max(deltas[s], abs(v - V[s]))

        #Use my final V to create policy.    
        for s in transitionTable.keys():
            policy[s], _ = bellmanUpdate(s, V, transitionTable, rewardTable, getSPrimeRDistributionFull, gamma, roundingTolerance)
            
        return policy


    policy = {}
    policy['max'] = valueIteration(V, transitionTable, rewardTable, convergenceTolerance, gamma, bellmanUpdate, getSPrimeRDistributionFull, roundingTolerance)

    hm.drawFinalMap(V, policy["max"], trapDict, bonusDict, blockList, normalCost)
    
    
    
if __name__=='__main__': 
    main()