import pandas as pd
import keras
from keras import layers
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


model = keras.models.load_model('./model.keras')

# Completions,PassAttempts,PassYD,PassTD,Int,RushYD,RushTD,PointsAgainst,XPA,XPM,FGA,FGM,SafetiesByTeam

ATL2019 = np.array(
    [[383, 574, 4606, 29, 10, 1788, 13, 405, 39, 38, 30, 24, 0]])


ARI2019 = np.array(
    [[311, 518, 3738, 21, 15, 1972, 11, 399, 30, 30, 27, 23, 0]])


BAL2019 = np.array(
    [[245, 415, 2934, 15, 13, 2897, 21, 281, 35, 35, 30, 26, 0]])


# Completions,PassAttempts,PassYD,PassTD,Int,RushYD,RushTD,PointsAgainst,XPA,XPM,FGA,FGM,SafetiesByTeam

BUF2019 = np.array(
    [[252, 454, 3043, 18, 14, 2181, 12, 344, 32, 32, 29, 25, 1]])


CAR2019 = np.array(
    [[335, 528, 3787, 24, 15, 1956, 15, 376, 39, 38, 32, 25, 0]])


CHI2019 = np.array(
    [[348, 555, 3910, 25, 14, 1985, 16, 277, 38, 38, 28, 24, 1]])


CIN2019 = np.array(
    [[350, 569, 4163, 28, 17, 1861, 12, 449, 29, 29, 27, 23, 0]])


CLE2019 = np.array(
    [[395, 587, 4814, 31, 16, 1956, 12, 378, 39, 39, 28, 24, 1]])


DAL2019 = np.array(
    [[362, 566, 4054, 24, 11, 2014, 16, 318, 34, 34, 28, 24, 0]])


DEN2019 = np.array(
    [[379, 614, 4116, 20, 15, 1843, 15, 341, 33, 33, 28, 24, 0]])


DET2019 = np.array(
    [[367, 566, 4041, 24, 15, 1893, 10, 342, 29, 28, 31, 25, 0]])


GB2019 = np.array([[373, 588, 4339, 30, 7, 1866, 13, 392, 32, 31, 31, 25, 0]])


HOU2019 = np.array(
    [[356, 545, 4401, 28, 12, 1801, 11, 316, 38, 37, 30, 24, 0]])


IND2019 = np.array(
    [[330, 544, 3820, 22, 10, 1973, 15, 318, 36, 35, 32, 25, 0]])


JAX2019 = np.array(
    [[360, 586, 3972, 23, 14, 1684, 11, 310, 32, 31, 28, 23, 2]])


KC2019 = np.array([[388, 590, 4933, 38, 13, 1995, 14, 407, 43, 42, 32, 25, 0]])


LAC2019 = np.array(
    [[375, 574, 4642, 31, 13, 1851, 13, 317, 37, 37, 30, 25, 0]])


LAR2019 = np.array(
    [[363, 579, 4769, 33, 13, 2096, 14, 376, 42, 41, 33, 26, 2]])


LV2019 = np.array([[378, 599, 4498, 25, 15, 1651, 12, 449, 32, 32, 31, 25, 0]])


MIA2019 = np.array(
    [[309, 530, 3531, 22, 18, 1770, 10, 415, 29, 29, 29, 24, 0]])


MIN2019 = np.array(
    [[365, 539, 3985, 27, 11, 1906, 11, 311, 37, 37, 30, 25, 0]])


NE2019 = np.array([[386, 592, 4517, 29, 10, 2019, 19, 325, 44, 45, 31, 25, 0]])
print(NE2019.shape)

# NE2019 = NE2019.reshape(1, -1)


NO2019 = np.array([[407, 574, 4596, 31, 10, 1998, 17, 347, 41, 41, 28, 24, 0]])


NYG2019 = np.array(
    [[371, 571, 4012, 22, 13, 1746, 12, 400, 33, 33, 30, 25, 0]])


NYJ2019 = np.array(
    [[342, 558, 3992, 24, 17, 1807, 11, 421, 25, 25, 30, 25, 0]])


PHI2019 = np.array(
    [[406, 633, 4729, 35, 14, 1857, 12, 342, 38, 38, 28, 24, 0]])


PIT2019 = np.array(
    [[407, 642, 4732, 31, 17, 1681, 12, 354, 40, 40, 26, 22, 2]])


SEA2019 = np.array(
    [[302, 479, 3723, 29, 10, 2410, 14, 335, 39, 39, 31, 26, 0]])


SF2019 = np.array([[404, 646, 5145, 29, 19, 1808, 11, 409, 40, 40, 30, 25, 1]])


TB2019 = np.array([[390, 612, 4805, 29, 19, 1724, 10, 438, 36, 35, 34, 26, 0]])


TEN2019 = np.array(
    [[312, 470, 3557, 20, 14, 2084, 16, 295, 36, 35, 30, 23, 0]])


WAS2019 = np.array(
    [[348, 593, 4184, 23, 17, 1930, 14, 341, 32, 31, 31, 25, 0]])


team_stats = [ARI2019, ATL2019, BUF2019, BAL2019, CAR2019, CHI2019, CIN2019, CLE2019, DAL2019, DEN2019, DET2019, GB2019, HOU2019, IND2019, JAX2019,
              KC2019, LAC2019, LAR2019, LV2019, MIA2019, MIN2019, NE2019, NO2019, NYG2019, NYJ2019, PHI2019, PIT2019, SEA2019, SF2019, TB2019, TEN2019, WAS2019]

team_wins = [5, 7, 10, 14, 5, 8, 2, 6, 8, 7, 3, 13, 10, 7, 6,
             12, 5, 9, 7, 5, 10, 12, 13, 4, 7, 9, 8, 11, 13, 7, 9, 3]

predicted_wins = []

betting_outcome = []

lines = []
lines.append({"line": 5, "over": -120, "under": 0})
lines.append({"line": 8.5, "over": -145, "under": 125})
lines.append({"line": 7, "over": -140, "under": 120})
lines.append({"line": 8.5, "over": -110, "under": -110})
lines.append({"line": 8, "over": -120, "under": 0})
lines.append({"line": 9, "over": -135, "under": 115})
lines.append({"line": 6, "over": 115, "under": -135})
lines.append({"line": 9, "over": -110, "under": -110})
lines.append({"line": 9, "over": -120, "under": 0})
lines.append({"line": 7, "over": -135, "under": 115})
lines.append({"line": 6.5, "over": -140, "under": 120})
lines.append({"line": 9, "over": -130, "under": 110})
lines.append({"line": 8.5, "over": 110, "under": -130})
lines.append({"line": 7.5, "over": -130, "under": 110})
lines.append({"line": 8, "over": -120, "under": 0})
lines.append({"line": 10.5, "over": -130, "under": 110})
lines.append({"line": 10, "over": 0, "under": -120})
lines.append({"line": 10.5, "over": 130, "under": -150})
lines.append({"line": 6, "over": -110, "under": -110})
lines.append({"line": 4.5, "over": -130, "under": 110})
lines.append({"line": 9, "over": -120, "under": 0})
lines.append({"line": 11, "over": -140, "under": 120})
lines.append({"line": 10.5, "over": 130, "under": -150})
lines.append({"line": 6, "over": 125, "under": -145})
lines.append({"line": 7.5, "over": -130, "under": 110})
lines.append({"line": 10, "over": -140, "under": 120})
lines.append({"line": 9, "over": -145, "under": 125})
lines.append({"line": 8.5, "over": -120, "under": 0})
lines.append({"line": 8, "over": -140, "under": 120})
lines.append({"line": 6.5, "over": -110, "under": -110})
lines.append({"line": 8, "over": 120, "under": -140})
lines.append({"line": 6, "over": 125, "under": -145})

bet_size = 1000

initial_budget = bet_size * 32
final_budget = initial_budget

def calculate_payout(wager, odds):
    if odds >= 0:
        payout = wager * (odds / 100 + 1)
    else:
        payout = wager * (100 / abs(odds) + 1)
    return payout

#bet_size = 10

for i in range(0, 32):
    prediction_func = model.predict(team_stats[i])
    predicted_wins_num = np.argmax(prediction_func)
    predicted_wins.append(predicted_wins_num)

    if (predicted_wins_num > lines[i].get("line")):
        #final_budget-=100
        difference = predicted_wins_num - lines[i].get("line")
        final_budget -= (bet_size * difference)
        if(predicted_wins_num > team_wins[i]):
            final_budget += calculate_payout(bet_size * difference, lines[i].get("over"))
    elif (predicted_wins_num < lines[i].get("line")):
        #final_budget-=100
        difference = lines[i].get("line") - predicted_wins_num
        final_budget -= (bet_size * difference)
        if(predicted_wins_num < team_wins[i]):
            final_budget += calculate_payout(bet_size * difference, lines[i].get("under"))

print(f"Initial Budget was {initial_budget}, Final Budget was {final_budget}, for a gain/loss of {final_budget - initial_budget} for a percent gain of {((final_budget-initial_budget)/initial_budget) * 100}")

