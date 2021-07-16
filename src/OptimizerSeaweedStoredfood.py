import numpy
import matplotlib.pyplot as plt

from pulp import LpMaximize, LpProblem, LpVariable, LpStatus

"""
In this code, you have to enter 
The Initial area in Km^2 / 10^4 and the new area available in line 16
The number of months to model line 21
The values for the different foods line 32 to 35
"""

def create_linear_model():
  # Stock of resources. Add the value of stored resources in bracket
  Area = {"Start": [0.1000], "End": [], "Production": [0.4153, 0.4153, 0.4153, 0.4153, 0.4153, 0.4153, 0.4153, 0.4153,
                                                     0.4153, 0.4153, 0.4153, 0.4153, 0.4153, 0.4153]}

  # Number of day for the experience
  Day = 14

  # Create the model to optimize
  model = LpProblem(name="optimization_nutrition", sense=LpMaximize)

  # Initialize the variable to maximize. That is, the maximal number of people we can feed each day
  z = LpVariable(name="minimum_value_nutrition", lowBound=0, cat='Integer')

  # Variables for food.
  # Calories per 10^4 Tons taken from nutrition calculator divided by 1'000'000'000 for rounding up errors
  # Proteins digestibility with same process as for calories
  # Fat total with same process as for calories
  S = {'Start': [], 'Eaten': [], 'After': [], 'Max_T_Area': 4000,
       'Calories': 29.000, 'Fat': 0.772, 'Proteins': 0.0852, 'Growing_period': 1}
  Stored_Wheat = {'Start': [], 'Eaten': [], 'After': [],
                  'Calories': 36.400, 'Fat': 0.098, 'Proteins': 0.097}

  # This part define one a variable for each type of food. Each food is a vector made of boxes which encode months
  for x in range(0, Day):
      S['Eaten'].append(LpVariable(name='Eaten_Seaweed%s' % x, lowBound=0))
      S['Start'].append(LpVariable(name='Stock_Seaweed%s' % x, lowBound=0))
      Stored_Wheat['Eaten'].append(LpVariable(name='Eaten_Stored_Wheat%s' % x, lowBound=0))
      Stored_Wheat['Start'].append(LpVariable(name='Stock_Stored_Wheat%s' % x, lowBound=0))
  S['Start'].append(LpVariable(name='Stock_Seaweed', lowBound=0))
  Stored_Wheat['Start'].append(LpVariable(name='Stock_Stored_Wheat', lowBound=0))
  # a,b,c are the variables for the calories, fat and proteins obtained by the food eaten
  # H encodes the number of people we can feed each month
  a = []
  b = []
  c = []
  H = []
  for x in range(0, Day):
      a.append(LpVariable(name='T_Calories_Eaten%s' % x, lowBound=0))
      b.append(LpVariable(name='T_Fat_Eaten%s' % x, lowBound=0))
      c.append(LpVariable(name='T_Proteins_Eaten%s' % x, lowBound=0))
      H.append(LpVariable(name='Humans_fed%s' % x, lowBound=0, cat='Integer'))

  Total_Eaten = {'Calories_Eaten': a, 'Fat_Eaten': b, 'Proteins_Eaten': c}

  model += (0.1 >= (S['Start'][0]), 'Seaweed_before_positive_start%s' % 0)
  model += (86200.0000 >= (Stored_Wheat['Start'][0]), 'Stored_before_positive_start%s' % 0)
  # Computation for months
  for x in range(0, Day):
      # Computation for seaweed: constraint for food eaten line 60 and the new quantity of seaweed at month x+1 is the
      # one at month x minus the seaweed eaten and then taking into account the reproduction
      S['After'].append(S['Start'][x] - S['Eaten'][x])
      model += (S['Start'][x+1] <= (S['Start'][x] - S['Eaten'][x]) * pow(1.1, 30) - 0.00000001,
                'Seaweed_positive_start%s' % x)

      # Computation for stored food: constraint for food eaten line 68 and the new quantity of stored wheat at month x+1
      # is the one at month x minus the wheat eaten
      Stored_Wheat['After'].append(Stored_Wheat['Start'][x] - Stored_Wheat['Eaten'][x])
      model += (Stored_Wheat['Start'][x + 1] <= (Stored_Wheat['Start'][x] - Stored_Wheat['Eaten'][x]),
                'Stored_positive_start%s' % x)

      # actualisation of available resources. At month x+1, we have the quantity of resources from month x minus the used
      # resources plus the production of resources per months
      Area['Start'].append(Area['Start'][x] + Area['Production'][x])
      model += (0 <= S['Max_T_Area'] * Area['Start'][x] - S['Start'][x], 'Area_End_constraint%s' % x)
      model += (100 >= Area['Start'][x], 'Area_Start_constraint%s' % x)
      # I assume a density of 4000 tons per Km^2 which is the maximum in the spreadsheet

      # Calories, fat and proteins obtained for the population by eating the stored wheat and the seaweeds
      Total_Eaten['Calories_Eaten'][x] = Stored_Wheat['Eaten'][x] * Stored_Wheat['Calories'] + \
                                       S['Calories'] * S['Eaten'][x]
      Total_Eaten['Fat_Eaten'][x] = Stored_Wheat['Eaten'][x] * Stored_Wheat['Fat'] + S['Fat'] * S['Eaten'][x]
      Total_Eaten['Proteins_Eaten'][x] = Stored_Wheat['Eaten'][x] * Stored_Wheat['Fat'] + S['Proteins'] * S['Eaten'][x]

      # Constraint for the number of people fed by the eaten foods
      model += (0 <= Total_Eaten['Calories_Eaten'][x] - H[x] * 7.1190 * pow(10, -5), 'Feeding_Calories%s' % x)
      # Numbers: 2100 Kcal/day * 1.13 for lost food * 30 for month/ 1'000'000'000
      model += (0 <= Total_Eaten['Fat_Eaten'][x] - H[x] * 2.373 * pow(10, -6), 'Feeding_Fat%s' % x)
      # Numbers:  70.00 g/day * 1.13 for lost food * 30 for month/ 1'000'000'000
      model += (0 <= Total_Eaten['Proteins_Eaten'][x] - H[x] * 2.669625 * pow(10, -6), 'Feeding_Proteins%s' % x)
      # Numbers:  78.75 g/day * 1.13 for lost food * 30 for month/ 1'000'000'000

  # Constraint for having minimum to be a linear function. We stored all variables and the number of people we can feed
  # each month. Now the final number of people we saved is the minimum other the number of people fed each month
  for x in range(Day):
      model += (H[x] >= z, 'Min_Nutrition%s' % x)

  # The model maximize the minimum an so maximize the number of people saved taking time into account
  obj_func = z
  model += obj_func
  
  return model
  
def run_linear():
  model = create_linear_model()
  status = model.solve()
  # print(LpStatus[status])  # Check if the solution is optimal

  # Print the different variables. That is, the number of people rescued and the different king of food we grew each month
  print(f"objective: {model.objective.value()}")
  for var in model.variables():
      print(f"{var.name}: {var.value()}")

  # Plot the diagram
  labels = []
  Stored_eaten = []
  Seaweed_eaten = []
  Stored_stock = [Stored_Wheat['Start'][0].value()]
  Seaweed_stock = [S['Start'][0].value()]
  Seaweed_stock_real = [S['Start'][0].value()]
  for x in range(0, Day):
      labels.append('Month%s' % x)
      Stored_eaten.append(Stored_Wheat['Eaten'][x].value())
      Seaweed_eaten.append(S['Eaten'][x].value())
  for x in range(0, Day-1):
      Seaweed_stock_real.append((Seaweed_stock_real[x] - S['Eaten'][x].value()) * pow(1.1, 30))
  for x in range(1, Day):
      Stored_stock.append(Stored_Wheat['Start'][x].value())
      Seaweed_stock.append(S['Start'][x].value())


  x = numpy.arange(len(labels))  # the label locations
  width = 0.20  # the width of the bars

  fig, ax = plt.subplots()
  Rectangles1 = ax.bar(x - 1.5 * width, Stored_eaten, width, label='Stored food eaten')
  Rectangles2 = ax.bar(x - 0.5 * width, Seaweed_eaten, width, label='Seaweeds eaten')
  Rectangles3 = ax.bar(x + 0.5 * width, Stored_stock, width, label='Stored food in stock')
  Rectangles4 = ax.bar(x + 1.5 * width, Seaweed_stock, width, label='Seaweeds in stock')
  # Rectangles5 = ax.bar(x + 1.5 * width, Seaweed_stock_real, width, label='Stored seaweed r')

  # Add some text for labels, title and custom x-axis tick labels, etc.
  ax.set_ylabel('10^4 Tons')
  ax.set_title('Food eaten and stocked over the months')
  ax.set_xticks(x)
  ax.set_xticklabels(labels)
  ax.legend()

  ax.bar_label(Rectangles1, padding=0)
  ax.bar_label(Rectangles2, padding=0)
  ax.bar_label(Rectangles3, padding=0)
  ax.bar_label(Rectangles4, padding=0)
  # ax.bar_label(Rectangles5, padding=0)

  fig.tight_layout()

  plt.show()

  Month_start_end_dates = [0]
  Area_months = [0.1 * 4000]
  for x in range(1, Day):
      print(Area['Start'][x])
      for y in range(0, 2):
          Month_start_end_dates.append(x)
          Area_months.append(Area['Start'][x] * 4000)

  Exact_computation = [0.1]
  for x in range(0, Day-1):
      Exact_computation.append(S['Start'][x].value() - S['Eaten'][x].value())
      Exact_computation.append(S['Start'][x+1].value())

  fig = plt.figure()
  plt.plot(Month_start_end_dates, Exact_computation)
  plt.plot(Month_start_end_dates, Area_months)

  plt.xlabel('Month')
  plt.ylabel('10^4 Tons')
  plt.title('Quantity of seaweeds vs Maximal quantity authorized by the area')
  plt.legend(['Seaweeds', 'Seaweeds accepted by the area'])
  plt.show()
 
if __name__ == '__main__':
    run_linear()
