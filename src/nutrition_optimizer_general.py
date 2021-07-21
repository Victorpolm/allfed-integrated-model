# Test if passing dividing every constant by the same amount reduces the problem of rounding numbers up
import numpy
import matplotlib.pyplot as plt

from pulp import LpMaximize, LpProblem, LpVariable, LpStatus
from matplotlib import pyplot

"""
In this code, you have to enter 
The stock of resources and their production line 17 to 21
The number of people the stock permits to feed for one month line 24
The number of months to model line 27
The calories, fat and proteins for 1 Ton of each food as well as their growing_period and the number of tons to harvest
per Km^2 line 36 to 41
"""

# Stock of resources. Add the value of stored resources in bracket
Pla = {"Start": [20], "End": [], "Production": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                                100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]}
A = [5.0 * pow(10, 8)]
FE = {"Start": [15], "End": [], "Production": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
                                               100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]}

# Choose the quantity of stored food in terms of number of people that can be feed for one month
Number_people_stock = 100

# Number of day for the experience
Day = 24

# Create the model to optimize
model = LpProblem(name="optimization_nutrition", sense=LpMaximize)

# Initialize the variable to maximize. That is, the maximal number of people we can feed each day
z = LpVariable(name="minimum_value_nutrition", lowBound=0, cat='Integer')

# Variables for food. with each nutriment per ton divided by 10'000 for rounding up errors (numbers are fake here)
G = {'Start': [], 'Planted': [], 'Eaten': [], 'After': [], 'Area': [0],
     'Calories': 0.500, 'Fat': 0.020, 'Proteins': 0.050, 'Growing_period': 2, 'KG_Area': 10}
R = {'Start': [], 'Planted': [], 'Eaten': [], 'After': [], 'Area': [0],
     'Calories': 2.000, 'Fat': 1.000, 'Proteins': 0.300, 'Growing_period': 2, 'KG_Area': 1}
C = {'Start': [], 'Planted': [], 'Eaten': [], 'After': [], 'Area': [0],
     'Calories': 1.556, 'Fat': 0.031, 'Proteins': 0.033, 'Growing_period': 4, 'KG_Area': 7}

# This part define one a variable for each type of food planted and eaten. Each food is a vector made of boxes which
# encode months
for x in range(0, Day):
    G['Planted'].append(LpVariable(name='Planted_Greenhouse_crops%s' % x, lowBound=0))
    G['Eaten'].append(LpVariable(name='Eaten_Greenhouse_crops%s' % x, lowBound=0))
    G['Start'].append(LpVariable(name='Start_Greenhouse_crops%s' % x, lowBound=0))
    R['Planted'].append(LpVariable(name='Planted_Ruminants%s' % x, lowBound=0))
    R['Eaten'].append(LpVariable(name='Eaten_Ruminants%s' % x, lowBound=0))
    R['Start'].append(LpVariable(name='Start_Ruminants%s' % x, lowBound=0))
    C['Planted'].append(LpVariable(name='Planted_Outdoor_crop%s' % x, lowBound=0))
    C['Eaten'].append(LpVariable(name='Eaten_Outdoor_crop%s' % x, lowBound=0))
    C['Start'].append(LpVariable(name='Start_Outdoor_crop%s' % x, lowBound=0))

G['Start'].append(LpVariable(name='Start_Greenhouse_crops', lowBound=0))
R['Start'].append(LpVariable(name='Start_Ruminants', lowBound=0))
C['Start'].append(LpVariable(name='Start_Outdoor_crop', lowBound=0))

# Initialize the variables for each type of food
# H encodes the number of people we can feed each month
a = []
b = []
c = []
d = []
e = []
f = []
H = []

for x in range(0, Day):
    a.append(LpVariable(name='Stored_Calories_Eaten%s' % x, lowBound=0))
    b.append(LpVariable(name='Stored_Fat_Eaten%s' % x, lowBound=0))
    c.append(LpVariable(name='Stored_Proteins_Eaten%s' % x, lowBound=0))
    d.append(LpVariable(name='T_Calories_Eaten%s' % x, lowBound=0))
    e.append(LpVariable(name='T_Fat_Eaten%s' % x, lowBound=0))
    f.append(LpVariable(name='T_Proteins_Eaten%s' % x, lowBound=0))
    H.append(LpVariable(name='Humans_fed%s' % x, lowBound=0, cat='Integer'))

# Calories per days * 30 for month and divided by 10'000 for rounding up errors
StoredF = {
    'Calories_Start': [], 'Fat_Start': [], 'Proteins_Start': [],
    'Calories_End': [], 'Fat_End': [], 'Proteins_End': [],
    'Calories_Eaten': a, 'Fat_Eaten': b, 'Proteins_Eaten': c
}

for x in range(0, Day):
    StoredF['Calories_Start'].append(LpVariable(name='Stored_calories%s' % x, lowBound=0))
    StoredF['Fat_Start'].append(LpVariable(name='Stored_fat%s' % x, lowBound=0))
    StoredF['Proteins_Start'].append(LpVariable(name='Stored_proteins%s' % x, lowBound=0))

StoredF['Calories_Start'].append(LpVariable(name='Stored_calories', lowBound=0))
StoredF['Fat_Start'].append(LpVariable(name='Stored_fat', lowBound=0))
StoredF['Proteins_Start'].append(LpVariable(name='Stored_proteins', lowBound=0))

Total_Eaten = {'Calories_Eaten': d, 'Fat_Eaten': e, 'Proteins_Eaten': f}

# Each food starts at 0 and stored food start with the number of people it can feed
model += (0 <= - G['Start'][0], 'Greenhouse_Start 0')
model += (0 <= - R['Start'][0], 'Ruminant_Start 0')
model += (0 <= - C['Start'][0], 'Crops_Start 0')
model += (StoredF['Calories_Start'][0] <= 7.1580 * Number_people_stock, 'Stored_calories_start')
model += (StoredF['Fat_Start'][0] <= 1.0737 * Number_people_stock, 'Stored_fat_start')
model += (StoredF['Proteins_Start'][0] <= 0.1800 * Number_people_stock, 'Stored_proteins_start')


# Computation for months
for x in range(0, Day):
    # Constraint resources: We cannot grow more than the current resources allow
    model += (0 <= Pla['Start'][x] - G['Planted'][x], 'Plastic_constraint%s' % x)
    model += (0 <= C['Start'][x] + G['Start'][x] - 7 * R['Planted'][x], 'food_constraint_ruminants%s' % x)
    model += (0 <= FE['Start'][x] - C['Planted'][x], 'Fertilizer_constraint%s' % x)

    # Computation of food eaten, at beginning and end of the month. if/else to take growing period into account
    if x != 0 and x % C['Growing_period'] == 0:
        C['After'].append(C['Start'][x] - C['Eaten'][x] + C['KG_Area']*C['Planted'][x - C['Growing_period']])
    else:
        C['After'].append(C['Start'][x] - C['Eaten'][x])
    C['Start'][x+1] = C['After'][x]
    # model += (0 <= C['Start'][x], 'Stored_Crops_is_positive%s' % x)

    if x != 0 and x % G['Growing_period'] == 0:
        G['After'].append(G['Start'][x] - G['Eaten'][x] + G['KG_Area']*G['Planted'][x - G['Growing_period']])
    else:
        G['After'].append(G['Start'][x] - G['Eaten'][x])
    G['Start'][x+1] = G['After'][x]
    # model += (0 <= G['Start'][x], 'Stored_Greenhouse_is_positive%s' % x)

    if x != 0 and x % R['Growing_period'] == 0:
        R['After'].append(R['Start'][x] - R['Eaten'][x] + R['KG_Area']*R['Planted'][x - R['Growing_period']])
    else:
        R['After'].append(R['Start'][x] - R['Eaten'][x])
    R['Start'][x+1] = R['After'][x]
    # model += (0 <= R['Start'][x], 'Stored_Ruminants_is_positive%s' % x)

    # Computation of stored food eaten, at beginning and end of the month for the 3 types of nutriments
    StoredF['Calories_End'].append(StoredF['Calories_Start'][x] - StoredF['Calories_Eaten'][x])
    model += (StoredF['Calories_Start'][x] <= StoredF['Calories_End'][x], 'Stored_Calories_start%s' % x)

    StoredF['Fat_End'].append(StoredF['Fat_Start'][x] - StoredF['Fat_Eaten'][x])
    model += (StoredF['Fat_Start'][x] <= StoredF['Fat_End'][x], 'Stored_Fat_start%s' % x)

    StoredF['Proteins_End'].append(StoredF['Proteins_Start'][x] - StoredF['Proteins_Eaten'][x])
    model += (StoredF['Proteins_Start'][x] <= StoredF['Proteins_End'][x], 'Stored_Proteins_start%s' % x)

    # Calories, fat and proteins obtained for the population by eating the stored wheat and the seaweeds
    Total_Eaten['Calories_Eaten'][x] = StoredF['Calories_Eaten'][x] + C['Calories'] * C['Eaten'][x] + \
                                       G['Calories'] * G['Eaten'][x] + R['Calories'] * R['Eaten'][x]
    Total_Eaten['Fat_Eaten'][x] = StoredF['Fat_Eaten'][x] + C['Fat'] * C['Eaten'][x] + \
                                       G['Fat'] * G['Eaten'][x] + R['Fat'] * R['Eaten'][x]
    Total_Eaten['Proteins_Eaten'][x] = StoredF['Proteins_Eaten'][x] + C['Proteins'] * C['Eaten'][x] + \
                                       G['Proteins'] * G['Eaten'][x] + R['Proteins'] * R['Eaten'][x]

    # Constraint for the number of people fed by the eaten foods
    model += (0 <= Total_Eaten['Calories_Eaten'][x] - H[x] * 7.1580, 'Feeding_Calories%s' % x)
    model += (0 <= Total_Eaten['Fat_Eaten'][x] - H[x] * 1.0737, 'Feeding_Fat%s' % x)
    model += (0 <= Total_Eaten['Proteins_Eaten'][x] - H[x] * 0.1800, 'Feeding_Proteins%s' % x)

    # Actualisation of available resources. At month x+1, we have the quantity of resources from month x minus the used
    # resources plus the production of resources per months
    Pla['End'].append(Pla['Start'][x] - G['Planted'][x] + Pla['Production'][x])
    Pla['Start'].append(Pla['End'][x])
    if x <= C['Growing_period']:
        C['Area'].append(C['Planted'][x])
    else:
        C['Area'].append(C['Area'][x] - C['Planted'][x] + C['Planted'][x - C['Growing_period']])
    if x <= G['Growing_period']:
        G['Area'].append(G['Planted'][x])
    else:
        G['Area'].append(G['Area'][x] - G['Planted'][x] + G['Planted'][x - G['Growing_period']])
    if x <= R['Growing_period']:
        R['Area'].append(R['Planted'][x])
    else:
        R['Area'].append(R['Area'][x] - R['Planted'][x] + R['Planted'][x - R['Growing_period']])
    A.append(5.0 * pow(10, 8) - C['Area'][x] - G['Area'][x] - R['Area'][x])
    model += (0 <= A[x], 'Area_constraint%s' % x)

    # No if/ else in this case since no growing period taken into account
    FE['End'].append(FE['Start'][x] - C['Planted'][x] - G['Planted'][x] + FE['Production'][x])
    FE['Start'].append(FE['End'][x])

# Constraint for having minimum to be a linear function. We stored all variables and the number of people we can feed
# each month. Now the final number of people we saved is the minimum other the number of people fed each month
for x in range(len(d)):
    model += (H[x] >= z, 'Min_Nutrition%s' % x)

# The model maximize the minimum an so maximize the number of people saved taking time into account
obj_func = z
model += obj_func
status = model.solve()
print(LpStatus[status])  # Check if the solution is optimal

# Print the different variables. That is, the number of people rescued and the different kind of food we grew each month
print(f"objective: {model.objective.value()}")
for var in model.variables():
    print(f"{var.name}: {var.value()}")

# Plot the diagram
labels = []
Calories_Eaten_StoredF = []
Calories_Eaten_Crops = []
Calories_Eaten_Greenhouse = []
Calories_Eaten_Ruminants = []
for x in range(0, Day):
    labels.append('Month%s' % x)
    Calories_Eaten_StoredF.append(StoredF['Calories_Eaten'][x].value() / 7.1580)
    Calories_Eaten_Crops.append((C['Eaten'][x].value() * C['Calories']) / 7.1580)
    Calories_Eaten_Greenhouse.append((G['Eaten'][x].value() * G['Calories']) / 7.1580)
    Calories_Eaten_Ruminants.append((R['Eaten'][x].value() * R['Calories']) / 7.1580)

Greenhouse = [G['Start'][0].value()]
Ruminant = [R['Start'][0].value()]
Crops = [C['Start'][0].value()]
Stored = [StoredF['Calories_Start'][0].value() / 7.1580]
for x in range(1, Day):
    Greenhouse.append(G['Start'][x].value())
    Ruminant.append(R['Start'][x].value())
    Crops.append(C['Start'][x].value())
    Stored.append(StoredF['Calories_Start'][x].value() / 7.1580)

x = numpy.arange(len(labels))  # the label locations
width = 0.125  # the width of the bars

fig, ax = plt.subplots()
Rectangles1 = ax.bar(x - 1.5 * width, Greenhouse, width, label='S')
Rectangles2 = ax.bar(x - 0.5 * width, Ruminant, width, label='C')
Rectangles3 = ax.bar(x + 0.5 * width, Crops, width, label='G')
Rectangles4 = ax.bar(x + 1.5 * width, Stored, width, label='R')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Number of people')
ax.set_title('Nutrition of stored food over the months')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(Rectangles1, padding=0)
ax.bar_label(Rectangles2, padding=0)
ax.bar_label(Rectangles3, padding=0)

fig.tight_layout()

plt.show()
