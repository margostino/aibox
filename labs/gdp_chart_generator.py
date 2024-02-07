import matplotlib.pyplot as plt

# UK\'s GDP data for the years 2020 to 2022

years = [2020, 2021, 2022]
gdp_values = [1.99, 2.14, 2.2]  # in trillion GBP

plt.figure(figsize=(10, 5))
plt.plot(years, gdp_values, marker="o")
plt.title("UK GDP from 2020 to 2022")
plt.xlabel("Year")
plt.ylabel("GDP (in trillion GBP)")
plt.grid(True)
plt.show()
