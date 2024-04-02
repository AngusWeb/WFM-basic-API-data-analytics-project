import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


# Load the data from the JSON file
with open("all_rivens_data.json", "r") as file:
    data = json.load(file)

# Access the riven auctions from the data
auctions = data["payload"]["auctions"]

# Create a list to store the extracted data
extracted_data = []

# Process each auction and extract relevant information
for auction in auctions:
    item = auction["item"]
    attributes = item["attributes"]
    
    # Create a dictionary for each auction with default values for attributes
    auction_data = {
        "starting_price": auction["starting_price"],
        "buyout_price": auction["buyout_price"],
        "top_bid": auction["top_bid"],
        "mastery_level": item["mastery_level"],
        "re_rolls": item["re_rolls"],
        "weapon_url_name": item["weapon_url_name"],
        "name": item["name"],
        "polarity": item["polarity"],
        "mod_rank": item["mod_rank"],
        "owner_reputation": auction["owner"]["reputation"],
        "owner_ingame_name": auction["owner"]["ingame_name"],
        "owner_status": auction["owner"]["status"],
        "platform": auction["platform"],
        "created": auction["created"],
        "updated": auction["updated"],
        "attribute_1": "",
        "1:value": float("nan"),
        "attribute_2": "",
        "2:value": float("nan"),
        "attribute_3": "",
        "3:value": float("nan"),
        "negative_attribute": "",
        "value": float("nan")
    }
    
    # Initialize counters for positive and negative attributes
    positive_count = 1
    negative_count = 1
    
    # Iterate over the attributes and assign them to the appropriate columns
    for attr in attributes:
        attr_name = attr["url_name"]
        attr_value = attr["value"]
        
        if attr["positive"]:
            if positive_count <= 3:
                auction_data[f"attribute_{positive_count}"] = attr_name
                auction_data[f"{positive_count}:value"] = attr_value
                positive_count += 1
        else:
            if negative_count == 1:
                auction_data["negative_attribute"] = attr_name
                auction_data["value"] = attr_value
                negative_count += 1
    
    extracted_data.append(auction_data)

# Create a DataFrame from the extracted data
vectisdf = pd.DataFrame(extracted_data)


# Total number of rivens
total_rivens = len(vectisdf)
print(f"Total number of rivens: {total_rivens}")

# Total sum of all buyouts
total_buyouts = vectisdf["buyout_price"].sum()
print(f"Total sum of all buyouts: {total_buyouts}")

# Total number of rerolls
total_rerolls = vectisdf["re_rolls"].sum()
print(f"Total number of rerolls: {total_rerolls}")

# Total unique weapon_url_name
unique_weapons = vectisdf["weapon_url_name"].nunique()
print(f"Total unique weapon_url_name: {unique_weapons}")



# Set the "top_bid" to be the "buyout_price" unless it's lower than "top_bid" * 1.5
vectisdf["top_bid"] = np.where(
    vectisdf["buyout_price"] >= vectisdf["top_bid"] * 1.5,
    vectisdf["buyout_price"],
    vectisdf["top_bid"]
)

vectisfullbo = vectisdf.copy()
# Remove entries where buyout_price is greater than 20000
vectisdf = vectisdf[vectisdf["buyout_price"] <= 20000]
# Reorder the columns
column_order = [
    "starting_price", "buyout_price", "mastery_level", "re_rolls", "weapon_url_name",
    "name", "polarity", "mod_rank", "owner_reputation", "owner_ingame_name",
    "owner_status", "platform", "created", "updated",
    "attribute_1", "1:value",
    "attribute_2", "2:value",
    "attribute_3", "3:value",
    "negative_attribute", "value"
]
vectisdf = vectisdf[column_order]



# Print the DataFrame
#print(vectisdf.head())



#print(vectisdf.describe())
#print(vectisdf.info())

# Select specific columns and view the top 10 values



attribute_mapping = {
    "ammo_maximum": "Ammo Maximum",
    "damage_vs_corpus": "Damage to Corpus",
    "damage_vs_grineer": "Damage to Grineer",
    "damage_vs_infested": "Damage to Infested",
    "cold_damage": "Cold",
    "channeling_damage": "Initial combo",
    "channeling_efficiency": "Heavy Attack Efficiency",
    "combo_duration": "Combo Duration",
    "critical_chance": "Critical Chance",
    "critical_chance_on_slide_attack": "Critical Chance for Slide Attack",
    "critical_damage": "Critical Damage",
    "base_damage_/_melee_damage": "Damage",
    "electric_damage": "Electricity",
    "heat_damage": "Heat",
    "finisher_damage": "Finisher Damage",
    "fire_rate_/_attack_speed": "Fire Rate / Attack Speed",
    "projectile_speed": "Projectile speed",
    "impact_damage": "Impact",
    "magazine_capacity": "Magazine Capacity",
    "multishot": "Multishot",
    "toxin_damage": "Toxin",
    "punch_through": "Punch Through",
    "puncture_damage": "Puncture",
    "reload_speed": "Reload Speed",
    "range": "Range",
    "slash_damage": "Slash",
    "status_chance": "Status Chance",
    "status_duration": "Status Duration",
    "recoil": "Weapon Recoil",
    "zoom": "Zoom",
    "chance_to_gain_extra_combo_count": "Additional Combo Count Chance",
    "chance_to_gain_combo_count": "Chance to Gain Combo Count"
}


# Create a new DataFrame with only the relevant columns
attribute_price_df = vectisdf[['buyout_price', 'attribute_1', 'attribute_2', 'attribute_3', 'negative_attribute']]

top_10_values = vectisdf.iloc[:10, [0, 1, 3, 5, 6, 14, 15, 16, 17, 18, 19, 20, 21]]

# Print the top 10 values with selected columns. head woud have been easier but was testing something before
print(top_10_values)


# Melt the DataFrame to combine attribute columns into a single column
melted_df = pd.melt(attribute_price_df, id_vars=['buyout_price'], value_vars=['attribute_1', 'attribute_2', 'attribute_3'], var_name='attribute', value_name='attribute_value')

# Filter out empty attribute values
melted_df = melted_df[melted_df['attribute_value'] != '']

# Map the attribute names using the attribute_mapping dictionary
melted_df['attribute_value'] = melted_df['attribute_value'].map(attribute_mapping)
attribute_price_df['negative_attribute'] = attribute_price_df['negative_attribute'].map(attribute_mapping)

# Calculate the average buyout price for each positive attribute
positive_attribute_impact = melted_df.groupby('attribute_value')['buyout_price'].mean()

# Calculate the average buyout price for each negative attribute
negative_attribute_impact = attribute_price_df.groupby('negative_attribute')['buyout_price'].mean()

# Create a table for positive attributes and sort it by average buyout price in descending order
positive_table = positive_attribute_impact.reset_index()
positive_table.columns = ['Positive Attribute', 'Average Buyout Price']
positive_table = positive_table.sort_values('Average Buyout Price', ascending=False)

# Create a table for negative attributes and sort it by average buyout price in descending order
negative_table = negative_attribute_impact.reset_index()
negative_table.columns = ['Negative Attribute', 'Average Buyout Price']
negative_table = negative_table.sort_values('Average Buyout Price', ascending=False)

""" # Print the tables
print("Positive Attribute Impact on Buyout Price:")
print(positive_table)
print("\nNegative Attribute Impact on Buyout Price:")
print(negative_table) """


# Set the color palette. ended up not using it
colors = ['#124455', '#d8f8e4']

# Create a bar graph for positive attributes
plt.figure(figsize=(10, 6))
bars = plt.bar(positive_table['Positive Attribute'], positive_table['Average Buyout Price'])
plt.xlabel('Positive Attribute', fontsize=12)
plt.ylabel('Average Buyout Price', fontsize=12)
plt.title('Impact of Positive Attributes on Buyout Price', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{round(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('positive_attributes_impact.png', dpi=300)
plt.show()

# Create a bar graph for negative attributes
plt.figure(figsize=(10, 6))
bars = plt.bar(negative_table['Negative Attribute'], negative_table['Average Buyout Price'])
plt.xlabel('Negative Attribute', fontsize=12)
plt.ylabel('Average Buyout Price', fontsize=12)
plt.title('Impact of Negative Attributes on Buyout Price', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{round(height)}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('negative_attributes_impact.png', dpi=300)
plt.show()

# Create a new DataFrame with only the relevant columns
attribute_price_df = vectisdf[['buyout_price', 'attribute_1', 'attribute_2', 'attribute_3', 'negative_attribute']]

# Function to determine the item type based on the number of positive and negative attributes
def get_item_type(row):
    positive_count = sum(1 for attr in [row['attribute_1'], row['attribute_2'], row['attribute_3']] if attr != '')
    negative_count = 1 if row['negative_attribute'] != '' else 0
    
    if positive_count == 2 and negative_count == 0:
        return '2 Positive, No Negative'
    elif positive_count == 3 and negative_count == 0:
        return '3 Positive, No Negative'
    elif positive_count == 2 and negative_count == 1:
        return '2 Positive, 1 Negative'
    elif positive_count == 3 and negative_count == 1:
        return '3 Positive, 1 Negative'
    else:
        return 'Other'

# Apply the get_item_type function to each row of the DataFrame
attribute_price_df['item_type'] = attribute_price_df.apply(get_item_type, axis=1)

# Calculate the count of each item type
item_type_counts = attribute_price_df['item_type'].value_counts()

# Create a bar graph for item type distribution
plt.figure(figsize=(8, 6))
plt.bar(item_type_counts.index, item_type_counts.values)
plt.xlabel('Item Type')
plt.ylabel('Count')
plt.title('Distribution of Item Types')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('item_type_distribution.png')
plt.show()

# Calculate the average buyout price for each item type
item_type_price = attribute_price_df.groupby('item_type')['buyout_price'].mean()
item_type_price = item_type_price.sort_values(ascending=False)

# Create a bar graph for item type and average buyout price
plt.figure(figsize=(8, 6))
plt.bar(item_type_price.index, item_type_price.values)
plt.xlabel('Item Type')
plt.ylabel('Average Buyout Price')
plt.title('Item Type vs Average Buyout Price')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('item_type_vs_average_buyout_price.png')
plt.show()

# Print the data as a table
print("Item Type vs Average Buyout Price")
print("----------------------------------")
print("{:<20} {:<20}".format("Item Type", "Average Buyout Price"))
print("----------------------------------")
for item_type, price in item_type_price.items():
    print("{:<20} {:<20.2f}".format(item_type, price))


# Remove null values and rivens worth more than 50000
vectisdf_filtered = vectisdf.dropna(subset=['re_rolls', 'buyout_price'])
vectisdf_filtered = vectisdf_filtered[vectisdf_filtered['buyout_price'] <= 10000]
#vectisdf_filtered = vectisdf_filtered[vectisdf_filtered['re_rolls'] > 0]

# Filter data to include only re-rolls between 0 and 1000
vectisdf_filtered = vectisdf_filtered[(vectisdf_filtered['re_rolls'] >= 0) & (vectisdf_filtered['re_rolls'] <= 1000)]

# Create a scatter plot of buyout_price vs re_rolls
plt.figure(figsize=(10, 6))
plt.scatter(vectisdf_filtered['re_rolls'], vectisdf_filtered['buyout_price'], s=3)

# Calculate the linear regression parameters
slope, intercept, r_value, p_value, std_err = stats.linregress(vectisdf_filtered['re_rolls'], vectisdf_filtered['buyout_price'])

# Generate points for the regression line
x_vals = np.linspace(vectisdf_filtered['re_rolls'].min(), vectisdf_filtered['re_rolls'].max(), 100)
y_vals = intercept + slope * x_vals

# Plot the regression line
plt.plot(x_vals, y_vals, color='red', label=f'Regression Line (R={r_value:.2f})')

plt.xlabel('Re-rolls')
plt.ylabel('Buyout Price')
plt.title('Buyout Price vs Re-rolls')

# Set the x-axis limit to start from 0 and end at 1000
plt.xlim(left=0, right=1000)

# Adjust the y-axis limit to start from 0
plt.ylim(bottom=0)

plt.legend()
plt.tight_layout()
plt.savefig('buyout_price_vs_re_rolls.png')
plt.show()

# Calculate the average buyout price for each polarity value
polarity_price = vectisdf.groupby('polarity')['buyout_price'].mean()
# Create a bar plot of average buyout price vs polarity
plt.figure(figsize=(7, 6))
plt.bar(polarity_price.index, polarity_price.values,)  
plt.xlabel('Polarity')
plt.ylabel('Average Buyout Price')
plt.title('Average Buyout Price vs Polarity')
plt.savefig('average_buyout_price_vs_polarity.png')
plt.tight_layout()
plt.show()


# Filter the DataFrame to include only rivens with weapon_url_name 'torid'
torid_df = vectisdf[vectisdf['weapon_url_name'] == 'torid']

# Function to check if a column contains the "critical_damage" word
def is_critical_damage(column_name):
    return "critical_damage" in column_name.lower()

# Function to get the critical damage value from the corresponding value column
def get_critical_damage(row):
    for i in range(1, 4):
        if is_critical_damage(row[f"attribute_{i}"]):
            return row[f"{i}:value"]
    if is_critical_damage(row["negative_attribute"]):
        return -row["value"]
    return None

# Apply the get_critical_damage function to each row of the filtered DataFrame
torid_df.loc[:, "critical_damage"] = torid_df.apply(get_critical_damage, axis=1)

# Apply the get_item_type function to each row of the filtered DataFrame
torid_df.loc[:, "item_type"] = torid_df.apply(get_item_type, axis=1)

# Filter the DataFrame to include only 3 positive 1 negative rivens
filtered_df = torid_df[torid_df["item_type"] == "3 Positive, 1 Negative"]

# Extract the non-null critical_damage values from the filtered DataFrame
critical_damage_values = filtered_df[filtered_df['critical_damage'].notnull()]['critical_damage']

# Convert the critical_damage values to numeric
critical_damage_values = pd.to_numeric(critical_damage_values, errors='coerce')

# Remove any NaN or invalid values
critical_damage_values = critical_damage_values[~np.isnan(critical_damage_values)]

# Check if there are any valid critical damage values
if len(critical_damage_values) > 0:
    # Define the bin edges for grouping the values into 5% increments
    bin_edges = np.arange(131.6, 160.9 + 5, 5)

    # Create the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(critical_damage_values, bins=bin_edges, edgecolor='black', align='mid', rwidth=0.8)

    # Set the x-axis tick labels to display the bin ranges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.xticks(bin_centers, [f'{bin_edges[i]:.1f}% - {bin_edges[i+1]:.1f}%' for i in range(len(bin_edges)-1)], rotation=45, ha='right')

    plt.xlabel('Critical Damage Range')
    plt.ylabel('Count')
    plt.title('Distribution of Critical Damage Values for 3 Positive 1 Negative Torid Rivens')
    plt.xlim(131.6, 160.9)  # Set the x-axis limits these values are from game wiki
    plt.tight_layout()
    plt.savefig('torid_critical_damage_distribution_3pos1neg.png')
    plt.show()
else:
    print("No valid critical damage values found for 3 Positive 1 Negative Torid rivens.")




# Create a graph comparing the frequency of entries across platforms
platform_counts = vectisdf['platform'].value_counts()

plt.figure(figsize=(10, 6))
plt.bar(platform_counts.index, platform_counts.values)
plt.xlabel('Platform')
plt.ylabel('Riven Count')
plt.title('Total Rivens by Platform')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the graph as an image file
plt.savefig('platform_frequency.png')
plt.show()
#Graphs dealing with demand riven ranking
# Graph 1: Number of rivens for each weapon_url_name (Top 20)
weapon_counts = vectisdf['weapon_url_name'].value_counts().nlargest(20)

plt.figure(figsize=(12, 6))
plt.bar(weapon_counts.index, weapon_counts.values)
plt.xlabel('Weapon URL Name')
plt.ylabel('Number of Rivens')
plt.title('Top 20 Weapons by Number of Rivens')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('riven_counts_per_weapon_top20.png')
plt.show()

# Graph 2: Number of rivens for each weapon_url_name (Top 100)
weapon_counts2 = vectisdf['weapon_url_name'].value_counts().nlargest(100)

plt.figure(figsize=(12, 6))
plt.bar(weapon_counts2.index, weapon_counts2.values)
plt.xlabel('Weapon URL Name')
plt.ylabel('Number of Rivens')
plt.title('Top 100 Weapons by Number of Rivens')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('riven_counts_per_weapon_top100.png')
plt.show()

# Exclude rivens with buyout price greater than 2000
vectisdf_filtered = vectisdf[vectisdf['buyout_price'] <= 20000]

# Graph 3: Average buyout price for each weapon_url_name (Top 20)
weapon_avg_buyout = vectisdf_filtered.groupby('weapon_url_name')['buyout_price'].mean().nlargest(20)

plt.figure(figsize=(18, 6))
plt.bar(weapon_avg_buyout.index, weapon_avg_buyout.values)
plt.xlabel('Weapon URL Name')
plt.ylabel('Average Buyout Price')
plt.title('Top 20 Weapons by Average Buyout Price (Buyout <= 20000)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('avg_buyout_per_weapon_top20_filtered.png')
plt.show()

# Graph 4: Average buyout price for each weapon_url_name (Top 50)
weapon_avg_buyout = vectisdf_filtered.groupby('weapon_url_name')['buyout_price'].mean().nlargest(50)

plt.figure(figsize=(14, 6))
plt.bar(weapon_avg_buyout.index, weapon_avg_buyout.values,)  # Adjust the width here
plt.xlabel('Weapon URL Name')
plt.ylabel('Average Buyout Price')
plt.title('Top 50 Weapons by Average Buyout Price (Buyout <= 20000)')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('avg_buyout_per_weapon_top50_filtered.png')
plt.show()

# Create a table of riven counts for each weapon_url_name, ranked from most to least
weapon_counts_table = vectisdf['weapon_url_name'].value_counts().reset_index()
weapon_counts_table.columns = ['Weapon URL Name', 'Number of Rivens']
weapon_counts_table['Rank'] = weapon_counts_table['Number of Rivens'].rank(method='dense', ascending=False).astype(int)

# Sort the table by the 'Number of Rivens' column in descending order
weapon_counts_table = weapon_counts_table.sort_values('Number of Rivens', ascending=False)

# Reset the index of the table
weapon_counts_table = weapon_counts_table.reset_index(drop=True)

# Print the table
print("Table: Number of Rivens for Each Weapon URL Name (Ranked)")
print(weapon_counts_table)

# Save the table to a CSV file
weapon_counts_table.to_csv('riven_counts_per_weapon_ranked.csv', index=False)

# Calculate the average buyout price for each weapon (Top 50)
top_50_avg_buyout = vectisdf_filtered.groupby('weapon_url_name')['buyout_price'].mean().nlargest(50)

# Calculate the number of rivens for each weapon (Top 50)
top_50_riven_counts = vectisdf['weapon_url_name'].value_counts().nlargest(50)

# Create a DataFrame to store the gaps
gap_df = pd.DataFrame({'Weapon URL Name': top_50_avg_buyout.index})
gap_df['Average Buyout Rank'] = gap_df.index + 1
gap_df['Riven Count Rank'] = gap_df['Weapon URL Name'].apply(lambda x: top_50_riven_counts.index.get_loc(x) + 1 if x in top_50_riven_counts.index else None)
gap_df['Rank Gap'] = gap_df['Riven Count Rank'] - gap_df['Average Buyout Rank']
gap_df['Average Buyout Price'] = gap_df['Weapon URL Name'].map(top_50_avg_buyout)
gap_df['Number of Rivens'] = gap_df['Weapon URL Name'].map(top_50_riven_counts)

# Sort the DataFrame by the 'Rank Gap' column in descending order
gap_df = gap_df.sort_values('Rank Gap', ascending=False)

# Print the table
print("Table: Weapons with the Biggest Gaps between Average Buyout Price and Number of Rivens Rankings")
print(gap_df)

# Save the table to a CSV file
gap_df.to_csv('weapon_rank_gaps.csv', index=False)

# Assuming your dataframe is named 'vectisdf' and follows the column order you provided
weapon = "zenith"  # Replace with the desired weapon URL name

# Filter the dataframe based on the specified criteria
filtered_df = vectisdf[
    (vectisdf["weapon_url_name"] == weapon) &
    (vectisdf["attribute_1"].isin(["critical_chance", "critical_damage", "multishot"])) &
    (vectisdf["attribute_2"].isin(["critical_chance", "critical_damage", "multishot"])) &
    (vectisdf["attribute_3"].isin(["critical_chance", "critical_damage", "multishot"])) &
    (vectisdf["negative_attribute"].notnull())
]

# Sort the filtered dataframe by buyout prices from high to low
sorted_df = filtered_df.sort_values("buyout_price", ascending=False)

# Display the sorted dataframe
print(sorted_df[column_order])

# Count the number of rivens for each polarity type
polarity_counts = vectisdf['polarity'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
plt.pie(polarity_counts, labels=polarity_counts.index, autopct='%1.1f%%')
plt.title('Distribution of Polarity Types')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.savefig('PolarityTypesPie.png')
plt.show()


#top_50_avg_buyout = vectisfullbo_filtered.groupby('weapon_url_name').apply(lambda x: x['buyout_price'].mean() - x.nsmallest(3, 'buyout_price')['buyout_price'].mean()).nlargest(50)

# Filter vectisfullbo to exclude rivens with less than 1 roll
vectisfullbo_filtered = vectisdf[vectisdf['re_rolls'] >= 1]

# Calculate the average buyout price for each weapon (Top 50)
top_50_avg_buyout = vectisfullbo_filtered.groupby('weapon_url_name')['buyout_price'].mean().nlargest(50)

# Calculate the number of rivens for each weapon (Top 150)
top_100_riven_counts = vectisfullbo_filtered['weapon_url_name'].value_counts().nlargest(100)

# Calculate the average buyout price of all rivens
overall_avg_buyout = vectisfullbo_filtered['buyout_price'].mean()

# Calculate the average frequency of all rivens
overall_avg_frequency = vectisfullbo_filtered['weapon_url_name'].value_counts().mean()

# Create a DataFrame to store the gaps and percentages
gap_df = pd.DataFrame({'Weapon URL Name': top_50_avg_buyout.index})
gap_df['Average Buyout Rank'] = gap_df.index + 1
gap_df['Riven Count Rank'] = gap_df['Weapon URL Name'].apply(lambda x: top_100_riven_counts.index.get_loc(x) + 1 if x in top_100_riven_counts.index else None)
gap_df['Rank Gap'] = gap_df['Riven Count Rank'] - gap_df['Average Buyout Rank']
gap_df['Average Buyout Price'] = gap_df['Weapon URL Name'].map(top_50_avg_buyout)
gap_df['Number of Rivens'] = gap_df['Weapon URL Name'].map(top_100_riven_counts)
gap_df['Percentage Above Average Price'] = ((gap_df['Average Buyout Price'] - overall_avg_buyout) / overall_avg_buyout) * 100
gap_df['Percentage Above Average Frequency'] = ((gap_df['Number of Rivens'] - overall_avg_frequency) / overall_avg_frequency) * 100

# Sort the DataFrame by the 'Rank Gap' column in descending order
gap_df = gap_df.sort_values('Rank Gap', ascending=False)

# Print the table
print("Table: Weapons with the Biggest Gaps between Average Buyout Price and Number of Rivens Rankings")
print(gap_df)

# Save the table to a CSV file
gap_df.to_csv('weapon_rank_gaps1.csv', index=False)

# Calculate the average buyout price for each weapon (Top 50)
top_50_avg_buyout = vectisfullbo_filtered.groupby('weapon_url_name')['buyout_price'].mean().nlargest(50)

# Calculate the number of rivens for each weapon (Top 150)
top_50_riven_counts = vectisfullbo_filtered['weapon_url_name'].value_counts().nlargest(50)

# Calculate the average buyout price of all rivens
overall_avg_buyout = vectisfullbo_filtered['buyout_price'].mean()

# Calculate the average frequency of all rivens
overall_avg_frequency = vectisfullbo_filtered['weapon_url_name'].value_counts().mean()

# Create a DataFrame to store the gaps and percentages
gap_df = pd.DataFrame({'Weapon URL Name': top_50_avg_buyout.index})
gap_df['Average Buyout Rank'] = gap_df.index + 1
gap_df['Riven Count Rank'] = gap_df['Weapon URL Name'].apply(lambda x: top_50_riven_counts.index.get_loc(x) + 1 if x in top_50_riven_counts.index else None)
gap_df['Rank Gap'] = gap_df['Riven Count Rank'] - gap_df['Average Buyout Rank']
gap_df['Average Buyout Price'] = gap_df['Weapon URL Name'].map(top_50_avg_buyout)
gap_df['Number of Rivens'] = gap_df['Weapon URL Name'].map(top_50_riven_counts)
gap_df['Percentage Above Average Price'] = ((gap_df['Average Buyout Price'] - overall_avg_buyout) / overall_avg_buyout) * 100
gap_df['Percentage Above Average Frequency'] = ((gap_df['Number of Rivens'] - overall_avg_frequency) / overall_avg_frequency) * 100

# Sort the DataFrame by the 'Rank Gap' column in descending order
gap_df = gap_df.sort_values('Rank Gap', ascending=False)

# Print the table
print("Table: Weapons with the Biggest Gaps between Average Buyout Price and Number of Rivens Rankings")
print(gap_df)

# Save the table to a CSV file
gap_df.to_csv('weapon_rank_gaps2.csv', index=False)