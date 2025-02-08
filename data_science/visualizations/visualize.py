import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./datasets/dataset_v1/states/World.csv")


# get years and population total
years = df[["Year"]]
population_total = df[["Population, total"]]

# input
known_index = int(len(years) * 0.8)
known_years = years.loc[0:known_index]
known_development = population_total.loc[0:known_index]

# output
out_years = years.loc[known_index:]
out_development = population_total.loc[known_index:]

plt.figure(figsize=(10, 6))
plt.plot(
    known_years,
    known_development,
    marker="o",
    linestyle="-",
    color="b",
    label="Predošlý vývoj",
)
plt.plot(
    out_years,
    out_development,
    marker="o",
    linestyle="-",
    color="r",
    label="Predpovedaný vývoj",
)

# Adding title and labels
plt.title("Svetová populácia", fontsize=14)
plt.xlabel("Rok", fontsize=12)
plt.ylabel("Celkový počet ľudí v populácii (v miliardách)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()
plt.tight_layout()

plt.savefig("work_goal.png")
