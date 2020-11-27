
def createRewardTable(transitionTable, normalCost, trapDict, bonusDict):
    rewardTable={s:{action:{sPrime:normalCost for sPrime in transitionTable[s][action].keys()} for action in transitionTable[s].keys()} for s in transitionTable.keys()}
    for s in rewardTable.keys():
        for a in rewardTable[s].keys():
            for sPrime in trapDict.keys():
                if rewardTable.get(s).get(a).get(sPrime) != None:
                    rewardTable[s][a][sPrime]=trapDict[sPrime]
            for sPrime in bonusDict.keys():
                if rewardTable.get(s).get(a).get(sPrime) != None:
                    rewardTable[s][a][sPrime]=bonusDict[sPrime]
    return rewardTable