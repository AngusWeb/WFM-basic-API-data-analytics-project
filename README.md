**Project Structure Overview**

This repository contains the code and resources for a data analysis project. Below is a breakdown of the key components and their locations within the project:

API Interaction Code: The logic for interacting with the API is encapsulated in the datawfm.py file. This script is responsible for fetching data from the API and preparing it for analysis.

Data Manipulation and Visualization: The datapandasall.py file contains the code for data manipulation, graphing, and tabulation using Pandas. This script transforms the raw data into insightful visualizations and tables.

Graphs: The visual output of the project, including all the graphs that were found to be useful, can be viewed in this image: [Graphs Image](https://i.imgur.com/pyggvAj.png). These graphs represent the analyzed data in a visual format, making it easier to understand and interpret the findings.

Data Extraction Scripts: Separate scripts were utilized to extract specific data from the dataframe, which was then used in API requests. These scripts are part of the data preprocessing steps that facilitate the analysis.

Failed Graph Attempts: It's worth noting that the project also includes several attempts at graphing that did not yield the expected results. These attempts are part of the iterative process of data analysis and visualization, showcasing the trial and error involved in finding the most effective ways to represent the data.


**Story of the Data**
[reddit post](https://en.reddit.com/r/Warframe/comments/1bu1ner/riven_guide_insights_from_warframemarket_data/)
I wanted to see if I could uncover some insights that would benefit both newbies and seasoned players, especially when it comes to rolling Rivens. I've put together some graphs that I think you'll find pretty interesting.

[Riven Results](https://i.imgur.com/pyggvAj.png)

All the data I used comes from a snapshot of warframe.market as of March 30, 2024. Here are some of the **questions** I explored:

- Which weapons have the most Rivens listed? And which ones have, on average, the priciest Rivens?
- Looking at the weapons, which ones show the biggest gap between the average price of Rivens and their ranking in terms of quantity? This could help veterans identify the most profitable Rivens to roll, indicating high demand but lower supply.
- How much pricier are Rivens with 3 positive and 1 negative trait compared to those with just 2 positive traits and no negatives?
- Is there a correlation between the number of rerolls on a Riven and its price?
- Do players pay attention to the polarity of Rivens?
- Which positive and negative stats on Rivens are considered the most valuable?

There are some statistics about the dataset:

- Total number of rivens: **193010**
- Total sum of all buyouts: 180936608p
- Total number of rerolls: 4658156     
- Total unique weapons: 385

Check out the table at the bottom of my image for weapons where Rivens are not only in high demand (aka $$$) but also have lower supply and fewer folks rolling them. 

**Based on the data, the most profitable Rivens to invest your Kuva in are:**

1. Magistar
2. Dual Ichor
3. Ceramic Dagger

