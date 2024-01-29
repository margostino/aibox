import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from pandasai.llm.openai import OpenAI
from pandasai import SmartDataframe
from pandasai import SmartDatalake

load_dotenv()

# create a dataframe of a csv from a URL https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/CO2%20emissions%20by%20sector%20(CAIT%2C%202020)/CO2%20emissions%20by%20sector%20(CAIT%2C%202020).csv
#CO2 emissions by sector (CAIT, 2020).csv
co2_emissions_by_sector = pd.read_csv('https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/CO2%20emissions%20by%20sector%20(CAIT%2C%202020)/CO2%20emissions%20by%20sector%20(CAIT%2C%202020).csv')
#print(df.head())
cancer_death_grouped = pd.read_csv('https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Cancer%20deaths%20grouped%20-%20OWID%20based%20on%20IHME/Cancer%20deaths%20grouped%20-%20OWID%20based%20on%20IHME.csv')


# df = pd.DataFrame({
#     "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
#     "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
#     "happiness_index": [6.94, 7.16, 6.66, 7.07, 6.38, 6.4, 7.23, 7.22, 5.87, 5.12]
# })

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
llm = OpenAI(api_token=OPENAI_API_KEY)
co2_emissions_by_sector = SmartDataframe(co2_emissions_by_sector, config={"llm": llm})
# reponse = co2_emissions_by_sector.chat('Which are the 5 top countries with most CO2 emissions?')
# print(reponse)
co2_emissions_by_sector.chat(
    "Plot the histogram of the top 5 countries with most CO2 emission, using different colors for each bar",
)


dl = SmartDatalake([co2_emissions_by_sector, cancer_death_grouped], config={"llm": llm})
co2_emissions_by_sector_description = """
CO2 emissions by sector (CAIT, 2020)
Carbon dioxide (CO₂) emissions broken down by sector, measured in tonnes per year. Further information on sector definitions is available here: https://ourworldindata.org/ghg-emissions-by-sector

This data is published by country and sector from the CAIT Climate Data Explorer, and downloaded from the Climate Watch Portal. Available here: https://www.climatewatchdata.org/data-explorer/historical-emissions
"""

cancer_death_grouped_description = """
Cancer deaths grouped - OWID based on IHME
Total annual number of deaths from cancers (termed 'Neoplasms' within the IHME, Global Burden of Disease Study). This measures cancer deaths across both sexes and all ages.

Smaller categories of cancer types have been grouped by Our World in Data into a collective category 'Other cancers'. This grouping was set based on cancer types with global annual deaths in 2016 under 100,000. This includes testicular, Hodgkin lymphoma, mesothelioma, thyroid, non-melanoma skin cancer, nasopharynx, malignant skin melanoma, uterine cancer, and multiple myeloma.
"""
query = f"""
Information about CO2 emission dataset: {co2_emissions_by_sector_description}
Information about Cancer death dataset: {cancer_death_grouped_description}
QUESTION: How CO2 data is correlated with Cancer death groups?
"""
response = dl.chat(f"""
ANSWER THIS QUESTION: How CO2 data is correlated with Cancer death groups?
Considering:
    - CO2 emission dataset description: {co2_emissions_by_sector_description}
    - Cancer death dataset description: {cancer_death_grouped_description}

""")
print(response)