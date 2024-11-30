# this is the program written by the model
from sympy import symbols, Eq, solve

# Define the cost variables
anyday_ticket_cost = 119
specific_day_ticket_cost = 79

# Define the number of people
num_people = 3

# Total cost if buying specific day tickets
specific_day_total_cost = specific_day_ticket_cost * num_people

# Total cost if buying anyday tickets with buy one get one offer
# We only need to buy 2 tickets because 1 will be free with the offer
anyday_total_cost = (num_people // 2 + num_people % 2) * anyday_ticket_cost

# Determine the minimum cost
min_cost = min(specific_day_total_cost, anyday_total_cost)

print(min_cost)
