from time import sleep
from colorama import Fore , Style
import random
from os import system

def show(s):
    global n
    print()
    for c in range(n+1):
        if(c == s):
            print(Fore.RED + str(c) + Style.RESET_ALL , end="")
            continue
        print(Fore.WHITE + str(c) + Style.RESET_ALL, end="")
    sleep(2)
    print()

def qeq(s1 , a , r , g , s2):
    return s1 + a*(r + g*s2 - s1)

def getmax(row):
    max = row[0]
    i = 0
    for c in range(0,3):
        if row[c] >= max:
            i = c
            max = row[c]
    return max , i


def train(state , qtable , a , g , n , e):
    goal = n if n%2 == 0 else n-1
    for c in range(0, e):
        s = state
        reward = [-1 , -1 , -50 , 100]
        moves = [-1 , 1 , 0 ]
        i = -1
        while i != 2:
            show(s)
            print(qtable)
            max , i = getmax(qtable[s])
            s2 = s + moves[i]
            if s2 < 0 or s2 > n:
                r = -25
                s2 = s
            else:
                    r = reward[i]
                    if s == goal  and i == 2:
                        r = reward[3]
                    elif s2 == goal :
                        r  = reward[3]/2
            max2 , i2 = getmax(qtable[s2])
            qtable[s][i] = qeq(qtable[s][i] , a , r , g , max2)
            s = s2

def test(qtable , init):
    global n
    s = init
    moves = [-1 , 1 , 0 ]
    i = -1
    while(i != 2):
       show(s)
       max , i = getmax(qtable[s])
       s2 = s + moves[i]
       if s2 < 0 or s2 > n:
                s2 = s
       s = s2

n = 10
qtable = [[-5.0, -3.3251814916966156, -10.0], [-3.0741391068048936, -0.5205511264926226, -10.0], [-2.692718858019898, 23.248983482744745, -10.0], [-2.392563811606113, 78.96636884077475, -10.0], [-2.0487527845685, 160.40296518179193, -10.0], [-1.942878157084454, 256.8322785020359, -10.0], [-1.5225751885762067, 360.7954528079317, -10.0], [-1.0881487994066843, 470.05709493950053, -10.0], [-0.5657996160000001, 585.0359665939603, -10.0], [0.0, 706.877426729861, -10.0], [0.0, 0.0, 780.2364424125608]]


test(qtable , 3)
print()
print(qtable)
